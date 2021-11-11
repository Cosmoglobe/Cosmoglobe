from typing import Dict, Protocol

from astropy.units import Quantity, Unit
import numpy as np

from cosmoglobe.sky._context_registry import ChainContextRegistry
from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.dust import ThermalDust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.radio import Radio
from cosmoglobe.sky.components.synchrotron import Synchrotron


class ChainContext(Protocol):
    """Protocol defining the interface for a chain context.

    Chain context defines additional processing required on the chain items
    before they are ready to be put into the sky model.

    A context needs to manipulate and return the `args` dictionary.
    
    NOTE: Context functions are executed *AFTER* renaming the units in the 
    chain. It is therefore important to use keys to the args dict that match
    the registered naming mappings.
    """

    def __call__(self, args: Dict[str, Quantity]) -> Dict[str, Quantity]:
        ...


def reshape_freq_ref(args: Dict[str, Quantity]) -> Dict[str, Quantity]:
    """Context that re-shapes the `freq_ref` attribute for unpolarized components."""

    if "freq_ref" in args:
        args["freq_ref"] = args["freq_ref"].to("GHz")
        if (amp_dim := args["amp"].shape[0]) == 1:
            args["freq_ref"] = args["freq_ref"][0].reshape((1, 1))
        elif amp_dim == 3:
            args["freq_ref"] = args["freq_ref"].reshape((3, 1))
        else:
            raise ValueError("cannot reshape freq_ref into shape (3,1) or (1,1")

    return args


def radip_specind(args: Dict[str, Quantity]) -> Dict[str, Quantity]:
    """Context that removes all but the first column of alpha.

    We are only interested in the column representinc the power law index
    of the spectral index paramter in the chainfiles.
    """

    args["alpha"] = args["alpha"][0]

    return args


def map_to_scalar(args: Dict[str, Quantity]) -> Dict[str, Quantity]:
    """Extract and returns a scalar.

    Datasets for unsamples quantities the cosmoglobe chains tends to be stored
    in HEALPIX maps. A quantity is considered a scalar if it is constant
    over all axes of the dataset.
    """

    IGNORED_ARGS = ["amp", "freq_ref"]
    for key, value in args.items():
        if key not in IGNORED_ARGS and np.size(value) > 1:
            if np.ndim(value) > 1:
                uniques = [np.unique(col) for col in value]
                all_cols_are_unique = all([len(col) == 1 for col in uniques])
            else:
                uniques = np.unique(value)
                all_cols_are_unique = len(uniques) == 1

            if all_cols_are_unique:
                args[key] = Quantity(uniques, unit=value.unit)

    return args


chain_context_registry = ChainContextRegistry()

chain_context_registry.register_context([], reshape_freq_ref)
chain_context_registry.register_context([], map_to_scalar)
chain_context_registry.register_context([Radio], radip_specind)

chain_context_registry.register_mapping([], {"freq_ref": "nu_ref"})
chain_context_registry.register_mapping([Radio], {"alpha": "specind"})
chain_context_registry.register_mapping([AME], {"freq_peak": "nu_p"})
chain_context_registry.register_mapping([FreeFree], {"T_e": "Te"})

chain_context_registry.register_units([], {"freq_ref": Unit("Hz")})
chain_context_registry.register_units(
    [AME, ThermalDust, Synchrotron, FreeFree], {"amp": Unit("uK_RJ")}
)
chain_context_registry.register_units([CMB], {"amp": Unit("uK_CMB")})
chain_context_registry.register_units([Radio], {"amp": Unit("mJy")})
chain_context_registry.register_units([AME], {"freq_peak": Unit("GHz")})
chain_context_registry.register_units([ThermalDust], {"T": Unit("K")})
chain_context_registry.register_units([FreeFree], {"T_e": Unit("K")})
