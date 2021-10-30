from typing import Optional, Union

from cosmoglobe.h5.chain import Chain
from cosmoglobe.refactored_sky.components import SkyComponent


def comp_from_chain(
    chain: Chain,
    nside: int,
    component: str,
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
                lmax = chain.get(f"{component}/{chain_arg}_lmax", samples=0)
                value = hp.alm2map(
                    value,
                    nside=nside if nside is not None else chain_params["nside"],
                    lmax=lmax,
                    fwhm=(chain_params["fwhm"] * u.arcmin).to("rad").value,
                    pol=pol,
                )

        args[arg] = u.Quantity(value, unit=units[arg] if arg in units else None)

    contexts = chain_context.get_context(component)
    for context in contexts:
        args.update(context(args))

    return comp_class(**args)
