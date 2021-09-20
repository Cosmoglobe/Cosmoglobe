from typing import Dict, Any, Protocol

import astropy.units as u
import numpy as np

from cosmoglobe.h5.chain_contextfactory import ChainContextFactory


class ChainContext(Protocol):
    """Protocol defining context for chain files.

    Chain context defines additional processing required on the chain items
    before they are ready to be put into the sky model.
    """

    def __call__(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Function that performs the processing on the specific chain item.

        This function needs to manipulate and return the `args` dictionary
        """


class FreqRefContext:
    """Re-shapes freq_ref for unpolarized components."""

    def __call__(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "freq_ref" in args:
            args["freq_ref"] = args["freq_ref"].to("GHz")
            if args["amp"].shape[0] != 3:
                args["freq_ref"] = args["freq_ref"][0]

        return args


class RadioContext:
    """Context for the radio component in the chain.

    In the chain, the amp of radio is stored in units of 'mJy'. Here we
    manually set convert it to 'uK' through astropy.

    NOTE: we pretend that amp has units of 'mJy/sr' to be able convert
    the units properly. This is fine since we always end dividing by
    the beam_area when converting the amplitudes to HEALPIX maps.
    """

    def __call__(self, args: Dict[str, Any]) -> Dict[str, Any]:
        args["alpha"] = args["alpha"][0]
        args["amp"] = u.Quantity(args["amp"].value, unit="mJy/sr").to(
            u.uK, equivalencies=u.thermodynamic_temperature(args["freq_ref"])
        )

        return args


class MapToScalarContext:
    """Extract and returns a scalar.

    Datasets in the cosmoglobe chains tends to be stored in HEALPIX maps.
    A quantity is considered a scalar if it is constant over all axes of the
    dataset.
    """

    def __call__(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in args.items():
            if np.size(value) > 1:
                if np.ndim(value) > 1:
                    uniques = [np.unique(col) for col in value]
                    all_cols_are_unique = all([len(col) == 1 for col in uniques])
                else:
                    uniques = np.unique(value)
                    all_cols_are_unique = len(uniques) == 1

                if all_cols_are_unique:
                    args[key] = u.Quantity(uniques, unit=value.unit)

        return args


chain_context = ChainContextFactory()

chain_context.register_context([], FreqRefContext)
chain_context.register_context([], MapToScalarContext)
chain_context.register_context(["radio"], RadioContext)

chain_context.register_mapping([], {"freq_ref": "nu_ref"})
chain_context.register_mapping(["radio"], {"alpha": "specind"})
chain_context.register_mapping(["ame"], {"freq_peak": "nu_p"})
chain_context.register_mapping(["ff"], {"T_e": "Te"})

chain_context.register_units([], {"amp": u.uK, "freq_ref": u.Hz})
chain_context.register_units(["ame"], {"nu_p": u.GHz})
chain_context.register_units(["dust"], {"T": u.K})
chain_context.register_units(["ff"], {"T_e": u.K})
