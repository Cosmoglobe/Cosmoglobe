from enum import Enum
from typing import Optional

from astropy.utils.data import download_file

from cosmoglobe.h5.chain import Chain

# from cosmoglobe.chain.h5 import model_from_h5

DATA_URL = "http://cosmoglobe.uio.no/BeyondPlanck/compsep/"


class Release(Enum):
    """Class that enumerates Cosmoglobe data releases."""

    BP10 = LATEST = "BP10"


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


# def sky_model(nside: int, release: Optional[str] = None, cache: bool = True) -> "Model":
#     """Initialize the Cosmoglobe Sky Model from a official release.

#     By default, the model is initialized from the latest stable cosmoglobe data
#     release. The data required to initialize the model is downloaded and cached
#     by default every time a unique nside is selected.

#     For a list of available releases see the get_releases function.

#     Parameters
#     ----------
#     nside : int
#         Healpix resolution parameter to initialize the model at.
#     release : int, str
#         Data release version. By default, the latest stable release is selected,
#         which corresponds to -1.
#         Default: -1
#     cache : bool
#         Wether to cache the data after the download. Default: True.
#     """

#     filename = f"model_{nside}.h5"
#     path = "/Users/metinsan/Documents/doktor/models/test1/"

#     if release is None:
#         release = Release.LATEST
#     else:
#         try:
#             release = Release[release.upper()]
#         except KeyError:
#             raise ValueError(
#                 f"Release {release} does not exist. Available releases are: "
#                 f"{[name for name, _ in Release.__members__.items() if name != 'LATEST']}"
#             )



if __name__ == "__main__":
    sky_model(64, "latesst")
