import inspect
import warnings
from typing import Dict, List, Literal, Optional, Type, Union

import healpy as hp
from astropy.units import Quantity, Unit
from tqdm import tqdm

from cosmoglobe.h5._exceptions import (
    ChainComponentNotFoundError,
    ChainFormatError,
    ChainKeyError,
)
from cosmoglobe.h5.chain import Chain, ChainVersion
from cosmoglobe.sky._base_components import (
    DiffuseComponent,
    PointSourceComponent,
    SkyComponent,
)
from cosmoglobe.sky._chain_context import chain_context_registry
from cosmoglobe.sky.cosmoglobe import DEFAULT_COSMOGLOBE_MODEL, CosmoglobeModel

DEFAULT_SAMPLE = -1


def get_components_from_chain(
    chain: Union[str, Chain],
    nside: int,
    components: Optional[List[str]] = None,
    model: CosmoglobeModel = DEFAULT_COSMOGLOBE_MODEL,
    samples: Optional[Union[range, int, Literal["all"]]] = DEFAULT_SAMPLE,
    burn_in: Optional[int] = None,
) -> Dict[str, SkyComponent]:
    """Initialize and return a cosmoglobe sky model from a chainfile.

    Parameters
    ----------
    chain
        Path to a Cosmoglobe chainfile or a Chain object.
    nside
        Model HEALPIX map resolution parameter.
    components
        List of components to include in the model.
    model
        String representing which sky model to use. Defaults to the
        BeyondPlanck model.
    samples
        The sample number for which to extract the model. If the input
        is 'all', then the model will an average of all samples in the chain.
        Defaults to the last sample in the chain.
    burn_in
        Burn in sample for which all previous samples are disregarded.

    Returns
    -------
    initialized_components
        List containing initialized sky components

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

    if ".astropy/cache" in str(chain.path):
        chain_name = "cached chainfile"
    else:
        chain_name = chain.path.name
    print(f"Initializing model from {chain_name}")
    if components is None:
        components = chain.components.copy()

        for component in chain.components:
            if component not in components:
                warnings.warn(
                    f"component {component} present in the Sky Model {model.version} "
                    "but not found in chain."
                )

    initialized_components: Dict[str, SkyComponent] = {}
    with tqdm(total=len(components), ncols=75) as progress_bar:
        padding = len(max(chain.components, key=len))
        for component in components:
            progress_bar.set_description(f"{component:<{padding}}")
            try:
                component_class = model[component]
            except KeyError:
                raise ChainComponentNotFoundError(
                    f"{component=!r} is not part in the Cosmoglobe Sky Model"
                )
            initialized_components[component] = get_comp_from_chain(
                chain, nside, component, component_class, samples
            )
            progress_bar.update()

    return initialized_components


def get_comp_from_chain(
    chain: Chain,
    nside: int,
    component_label: str,
    component_class: Type[SkyComponent],
    samples: Optional[Union[range, int]] = None,
) -> SkyComponent:
    """Initialize and return a sky component from a chainfile.

    Parameters
    ----------
    chain
        Chain object.
    nside
        Model HEALPIX map resolution parameter.
    component_label
        Name of the component.
    component_class
        Class representing the sky component.
    samples
        A range object, or an int representing the samples to use from the
        chain.

    Returns
    -------
        Initialized sky component object.
    """

    class_args = get_comp_signature(component_class)
    args: Dict[str, Quantity] = {}

    # Chain contexts are operations that we perform on the data in the chain
    # to fit it to the format required in the Cosmoglobe Sky Model. This includes,
    # renaming of variables (mappings), specifying astropy units, and reshaping
    # and/or converting maps constant over the sky to scalars.
    parameter_mappings = chain_context_registry.get_parameter_mappings(component_class)
    units = chain_context_registry.get_units(component_class)
    for arg in class_args:
        chain_arg = parameter_mappings.get(arg, arg)
        chain_params = chain.parameters[component_label]

        if chain_arg in chain_params:
            value = chain_params[chain_arg]
        else:
            try:
                value = chain.mean(
                    f"{component_label}/{chain_arg}_alm", samples=samples
                )
                is_alm = True
            except ChainKeyError:
                try:
                    value = chain.mean(
                        f"{component_label}/{chain_arg}", samples=samples
                    )
                except ChainKeyError:
                    value = chain.mean(
                        f"{component_label}/{chain_arg}_map", samples=samples
                    )
                is_alm = False

            if is_alm:
                pol = True if arg == "amp" and value.shape[0] == 3 else False
                lmax = chain.get(f"{component_label}/{chain_arg}_lmax", samples=0)
                value = hp.alm2map(
                    value,
                    nside=nside if nside is not None else chain_params["nside"],
                    lmax=lmax,
                    fwhm=(chain_params["fwhm"] * Unit("arcmin")).to("rad").value,
                    pol=pol,
                )

        args[arg] = Quantity(value, unit=units[arg] if arg in units else None)

    functions = chain_context_registry.get_functions(component_class)
    for function in functions:
        args.update(function(args))

    return component_class(**args)


def get_comp_signature(comp_class: Type[SkyComponent]) -> List[str]:
    """Extracts and returns the parameters to extract from the chain."""

    EXCLUDE = [
        "self",
        "freqs",
        "spectral_parameters",
        "_",
        "__",
    ]
    arguments: List[str] = []
    class_signature = inspect.signature(comp_class)
    arguments.extend(list(class_signature.parameters.keys()))
    if issubclass(comp_class, (DiffuseComponent, PointSourceComponent)):
        SED_signature = inspect.signature((comp_class.get_freq_scaling))
        arguments.extend(list(SED_signature.parameters.keys()))

    arguments = [arg for arg in arguments if arg not in EXCLUDE]

    return arguments
