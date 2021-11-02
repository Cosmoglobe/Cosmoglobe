from astropy.utils.data import download_file

from cosmoglobe.h5.chain import Chain


DATA_URL = "http://cosmoglobe.uio.no/BeyondPlanck/compsep/"


def get_test_chain(cache: bool = True) -> Chain:
    """Returns a minimal Cosmoglobe chain.

    This function will download the chain if it is not already cached.
    The chain is of size 1.6 GB.
    
    Parameters
    ----------
    cache
        Boolean for wether or not to cache the downloaded chain. Defaults to
        True
    
    Returns
    -------
    Chain
        Initialized Chainfile object.
    """

    filename = DATA_URL + "compsep_chain.h5"
    file = download_file(filename, cache=cache)

    return Chain(file)
