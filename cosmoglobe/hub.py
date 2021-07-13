from cosmoglobe.chain.h5 import model_from_h5

from astropy.utils.data import download_file

data_url = 'http://cosmoglobe.uio.no/BeyondPlanck/precomputed/'
   

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
    #download h5 file from the cosmoglobe web and cache

    path = '/Users/metinsan/Documents/doktor/models/test1/'
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