import inspect
from typing import Any, Optional, Union

import astropy.units as u
import healpy as hp
import numpy as np
from tqdm import tqdm

from cosmoglobe.h5.chain import Chain
from cosmoglobe.h5.exceptions import ChainComponentNotFoundError, ChainItemNotFoundError
from cosmoglobe.sky import COSMOGLOBE_COMPS
from cosmoglobe.sky.base import SkyComponent
from cosmoglobe.sky.model import Model


# These dicts needs to be manualy updated with new cosmoglobe components.
# Differences in argument names between the cosmoglobe sky component classes
# and commander chain outputs.
ARG_MAPPING = {
    "freq_ref": "nu_ref",
}
ARG_UNITS = {
    "amp": u.K,
    "freq_ref": u.Hz,
    "nu_p": u.GHz,
    "T": u.K,
    "Te": u.K,
}


def model_from_chain(
    chain: Union[str, Chain], nside: Optional[int] = None, burn_in: Optional[int] = None
) -> Model:
    """Initialize and return a cosmoglobe sky model from a chainfile.

    Parameters
    ----------
    chain
        Path to a Cosmoglobe chainfile or a Chain object.
    nside
        Model HEALPIX map resolution parameter.
    burn_in
        Burn in sample in the chain.

    Returns
    -------
    model
        Initialized sky mode object.
    """

    if not isinstance(chain, Chain):
        chain = Chain(chain, burn_in)

    model = Model(nside=nside)
    with tqdm(total=len(chain.components)) as progress_bar:
        padding = len(max(chain.components, key=len))
        for component in chain.components:
            progress_bar.set_description(f"{component:<{padding}}")
            initialized_component = component_from_chain(component, chain, nside)
            model._add_component_to_model(initialized_component)
            progress_bar.update()

    return model


def component_from_chain(
    component: str, chain: Chain, nside: Optional[int] = None
) -> SkyComponent:
    """Initialize and return a sky component from a chainfile.

    Parameters
    ----------
    component
        Name of the component in the chain.
    chain
        Chain object.
    nside
        Model HEALPIX map resolution parameter.

    Returns
    -------
        Initialized sky component object.
    """

    try:
        comp_class = COSMOGLOBE_COMPS[component]
    except KeyError:
        raise ChainComponentNotFoundError(
            f"Component {component} is not present in the chain"
        )
    signature = inspect.signature(comp_class)
    class_args = list(signature.parameters.keys())

    args = {}
    for arg in class_args:
        chain_arg = ARG_MAPPING.get(arg, arg)
        chain_params = chain.parameters[component]

        if chain_arg in chain_params:
            value = chain_params[chain_arg]
        else:
            try:
                value = chain.mean(f"{component}/{arg}_alm")
                is_alm = True
            except ChainItemNotFoundError:
                try:
                    value = chain.mean(f"{component}/{arg}")
                except ChainItemNotFoundError:
                    value = chain.mean(f"{component}/{arg}_map")
                is_alm = False

            if is_alm:
                pol = True if arg == "amp" and value.shape[0] == 3 else False
                value = hp.alm2map(
                    value,
                    nside=nside if nside is not None else chain_params["nside"],
                    fwhm=(chain_params["fwhm"] * u.arcmin).to("rad").value,
                    pol=pol,
                )

            # Amplitudes needs to be in map format in the sky model
            if "amp" not in arg:
                value = extract_scalar(value)

        args[arg] = u.Quantity(value, unit=ARG_UNITS[arg] if arg in ARG_UNITS else None)

    if "freq_ref" in args:
        args["freq_ref"] = args["freq_ref"].to("GHz")
        if chain.parameters[component]["polarization"].lower() == 'false':
            args["freq_ref"] = args["freq_ref"][0]

    return comp_class(**args)


def extract_scalar(value: Any) -> Any:
    """Extract and returns a scalar.

    Datasets in the cosmoglobe chains tends to be stored in HEALPIX maps.
    A quantity is considered a scalar if it is constant in over all axes.
    """

    if not np.size(value) > 1:
        return value

    if np.ndim(value) > 1:
        uniques = [np.unique(col) for col in value]
        all_cols_are_unique = all([len(col) == 1 for col in uniques])
    else:
        uniques = np.unique(value)
        all_cols_are_unique = len(uniques) == 1

    if all_cols_are_unique:
        return np.asarray(uniques)

    return value
