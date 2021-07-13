import pathlib
from healpy.pixelfunc import get_nside
from matplotlib.pyplot import show
from cosmoglobe.chain.h5 import model_from_h5

from astropy.utils.data import download_file

data_url = pathlib.Path('http://cosmoglobe.uio.no/')
   

def skymodel(nside, release=-1, cache=True):
    """Initialize the Cosmoglobe Sky Model from a official release.

    By default, the model is initialized from the latest stable commander3 
    release. The data required to initialize the model is downloaded and cached
    by default every time a unique nside is selected.

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
        release = releases[-1]
    else:
        if release not in releases:
            raise ValueError('Invalid release')

    path_to_h5 = data_url / release / filename

    # h5_file = download_file(path_to_h5, cache=cache, show_progress=True)
    #download h5 file from the cosmoglobe web and cache
    #     
    # model = model_from_h5(h5_file)
    model = model_from_h5(path+filename)
    return model


def get_releases():
    """Returns a list of stable Cosmoglobe data release versions. These can be
    used as input to `cosmoglobe.hub.skymodel`.
    """

    stable_releases = [
        'BP8',
        'BP9',
    ]
    return stable_releases