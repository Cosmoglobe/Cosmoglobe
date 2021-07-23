import pathlib
from enum import Enum

from astropy.utils.data import download_file

from cosmoglobe.chain.h5 import model_from_h5

data_url = pathlib.Path('http://cosmoglobe.uio.no/releases/')
   
class Release(Enum):
    """Class that enumerates Cosmoglobe data releases."""

    BP9 = LATEST = 'BP9/'
    BP8 = 'BP8'
    BP7 = 'BP7'


releases = {
    f'{release.name}'.lower(): release for release in Release
}


def sky_model(nside, release=-1, cache=True):
    """Initialize the Cosmoglobe Sky Model from a official release.

    By default, the model is initialized from the latest stable cosmoglobe data 
    release. The data required to initialize the model is downloaded and cached
    by default every time a unique nside is selected.

    For a list of available releases see the get_releases function.

    Parameters
    ----------
    nside : int
        Healpix resolution parameter to initialize the model at.
    release : int, str
        Data release version. By default, the latest stable release is selected,
        which corresponds to -1.
        Default: -1
    cache : bool
        Wether to cache the data after the download. Default: True.
    """

    filename = f'model_{nside}.h5'
    path = '/Users/metinsan/Documents/doktor/models/test1/'

    # select latest
    releases = get_releases()
    if release == -1:
        release = Release.LATEST
    else:
        if release not in releases:
            raise ValueError('Invalid release')

    path_to_h5 = data_url / release.value / filename

    # h5_file = download_file(path_to_h5, cache=cache, show_progress=True)
    #download h5 file from the cosmoglobe web and cache
    #     
    # model = model_from_h5(h5_file)
    model = model_from_h5(path+filename)
    return model


def get_releases():
    """Returns a list of stable Cosmoglobe data release versions."""
    
    return list(releases.keys())