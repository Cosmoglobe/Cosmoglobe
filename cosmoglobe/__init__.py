from typing import List, Literal, Optional, Union

from astropy.utils.data import download_file

from cosmoglobe.h5.chain import Chain
from cosmoglobe.plot import gnom, hist, plot, spec, standalone_colorbar, trace
from cosmoglobe.sky._units import K_CMB, K_RJ
from cosmoglobe.sky.model import SkyModel

_COMPSEP_DATA_URL = "http://sdc.uio.no/vol/cosmoglobe-data/BeyondPlanck/compsep/"
_BP_DATA_URL = "http://sdc.uio.no/vol/cosmoglobe-data/BeyondPlanck/releases/v2/"

def get_test_chain(cache: bool = True) -> Chain:
    """Returns a minimal Cosmoglobe chain.

    This function will download the chain if it is not already cached.
    The chain is of size 1.6 GB.
    
    Parameters
    ----------
    cache
        Boolean for wether or not to cache the downloaded chain. Defaults to True.
    
    Returns
    -------
    Chain
        Initialized Chainfile object.
    """

    filename = _COMPSEP_DATA_URL + "compsep_chain.h5"
    file = download_file(filename, cache=cache)

    return Chain(file)

def sky_model_from_chain(
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


def sky_model(nside: int, cache: bool = True) -> SkyModel:
    """Downloads and caches the Cosmoglobe Sky model. 
    
    The downloaded file is ~700 MB and contains the mean of all parameters
    from all BP production chains.

    Parameters
    ----------
    nside
        Healpix resolution of the model.

    cache
        Boolean for wether or not to cache the downloaded chain. Defaults to True.
    
    Returns
    -------
        Cosmoglobe `SkyModel`.
    """

    filename = _BP_DATA_URL + "BP_mean_v2.h5"
    file = download_file(filename, cache=cache)

    return SkyModel.from_chain(
        chain=file,
        nside=nside,
    )

__all__ = [
    "sky_model", 
    "get_test_chain", 
    "sky_model_from_chain", 
    "plot", 
    "spec", 
    "hist", 
    "gnom", 
    "trace", 
    "standalone_colorbar",
    "K_RJ",
    "K_CMB",
]