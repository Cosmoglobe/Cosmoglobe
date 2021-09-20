import inspect
from typing import List, Union, Optional, Literal

import astropy.units as u
import healpy as hp
from tqdm import tqdm

from cosmoglobe.h5.chain import Chain, ChainVersion
from cosmoglobe.h5.chain_context import chain_context
from cosmoglobe.h5.exceptions import (
    ChainComponentNotFoundError,
    ChainKeyError,
    ChainFormatError,
)
from cosmoglobe.sky.model import Model
from cosmoglobe.sky.base import SkyComponent
from cosmoglobe.sky import COSMOGLOBE_COMPS


DEFAULT_SAMPLE = -1


def model_from_chain(
    chain: Union[str, Chain],
    components: Optional[List[str]] = None,
    nside: Optional[int] = None,
    samples: Optional[Union[range, int, Literal["all"]]] = DEFAULT_SAMPLE,
    burn_in: Optional[int] = None,
) -> Model:
    """Initialize and return a cosmoglobe sky model from a chainfile.

    Parameters
    ----------
    chain
        Path to a Cosmoglobe chainfile or a Chain object.
    components
        List of components to include in the model.
    nside
        Model HEALPIX map resolution parameter.
    samples
        The sample number for which to extract the model. If the input
        is 'all', then the model will an average of all samples in the chain.
        Defaults to the last sample in the chain.
    burn_in
        Burn in sample for which all previous samples are disregarded.

    Returns
    -------
    model
        Initialized sky mode object.
    """

    if not isinstance(chain, Chain):
        chain = Chain(chain, burn_in)

    if isinstance(samples, str):
        if samples.lower() == "all":
            samples = None
        else:
            raise ValueError("samples must be either 'all', an int, or a range")

    if chain.version is ChainVersion.OLD:
        raise ChainFormatError(
            "cannot initialize a sky model from a chain without a " "parameter group"
        )

    print(f"Initializing model from {chain.path.name}")
    if components is None:
        components = chain.components
    elif any(component not in chain.components for component in components):
        raise ChainComponentNotFoundError(f"component was not found in chain")

    initialized_components: List[SkyComponent] = []
    with tqdm(total=len(components), ncols=75) as progress_bar:
        padding = len(max(chain.components, key=len))
        for component in components:
            progress_bar.set_description(f"{component:<{padding}}")
            initialized_components.append(
                _comp_from_chain(chain, component, nside, samples)
            )
            progress_bar.update()

    return Model(nside=nside, components=initialized_components)


def _comp_from_chain(
    chain: Chain,
    component: str,
    nside: Optional[int] = None,
    samples: Optional[Union[range, int]] = None,
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
            f"{component=!r} is not part in the Cosmoglobe Sky Model"
        )

    signature = inspect.signature(comp_class)
    class_args = list(signature.parameters.keys())

    args = {}

    # Contexts are operations that needs to be done to the data in the
    # chain before it can be used in the sky model.
    mappings = chain_context.get_mappings(component)
    units = chain_context.get_units(component)

    for arg in class_args:
        chain_arg = mappings.get(arg, arg)
        chain_params = chain.parameters[component]

        if chain_arg in chain_params:
            value = chain_params[chain_arg]
        else:
            try:
                value = chain.mean(f"{component}/{chain_arg}_alm", samples=samples)
                is_alm = True
            except ChainKeyError:
                try:
                    value = chain.mean(f"{component}/{chain_arg}", samples=samples)
                except ChainKeyError:
                    value = chain.mean(f"{component}/{chain_arg}_map", samples=samples)
                is_alm = False

            if is_alm:
                pol = True if arg == "amp" and value.shape[0] == 3 else False
                value = hp.alm2map(
                    value,
                    nside=nside if nside is not None else chain_params["nside"],
                    fwhm=(chain_params["fwhm"] * u.arcmin).to("rad").value,
                    pol=pol,
                    pixwin=True,
                )

        args[arg] = u.Quantity(value, unit=units[arg] if arg in units else None)

    contexts = chain_context.get_context(component)
    for context in contexts:
        args.update(context(args))

    return comp_class(**args)
