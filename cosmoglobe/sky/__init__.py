from typing import List, Literal, Optional, Union

from cosmoglobe.h5.chain import Chain
from cosmoglobe.sky._units import *
from cosmoglobe.sky.model import SkyModel


def model_from_chain(
    chain: Union[str, Chain],
    nside: int,
    components: Optional[List[str]] = None,
    model: str = "BeyondPlanck",
    samples: Optional[Union[range, int, Literal["all"]]] = -1,
    burn_in: Optional[int] = None,
):
    """Returns a SkyModel initialized from the chain.

    Parameters
    ----------
    chain
        Path to a Cosmoglobe chainfile or a `Chain`.
    nside
        Model HEALPIX map resolution parameter.
    components
        List of components to include in the model.
    model
        String representing which Cosmoglobe model to use. Defaults to
        BeyondPlanck.
    samples
        The sample number for which to extract the model. If the input
        is 'all', then the model will be initialized with the average of all
        samples in the chain. Defaults to the last sample in the chain (-1).
    burn_in
        If samples is 'all', a burn_in sample can be provided for which all
        subsequent samples are used in the averaging.

    Returns
    -------
        A SkyModel initialized from the chain.

    Examples
    --------
    >>> from cosmoglobe.sky import model_from_chain
    >>> sky_model = model_from_chain("path/to/chainfile.h5", nside=256)
    >>> print(sky_model)
    SkyModel(
        version: BeyondPlanck
        nside: 256
        components(
            (ame): AME(freq_peak)
            (cmb): CMB()
            (dust): ThermalDust(beta, T)
            (ff): FreeFree(T_e)
            (radio): Radio(alpha)
            (synch): Synchrotron(beta)
        )
    )
    """

    return SkyModel.from_chain(
        chain=chain,
        nside=nside,
        components=components,
        model=model,
        samples=samples,
        burn_in=burn_in,
    )


def skymodel(
    nside: int,
    components: Optional[List[str]] = None,
    model: str = "BeyondPlanck",
):
    """Returns a SkyModel from the hub."""

    raise NotImplementedError
