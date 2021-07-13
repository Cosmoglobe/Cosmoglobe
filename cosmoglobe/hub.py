from cosmoglobe.sky.model import Model
from cosmoglobe.sky import components

from astropy.utils.data import download_file

data_url = 'http://cosmoglobe.uio.no/BeyondPlanck/precomputed/'

COSMOGLOBE_COMPS = dict(
    ame=components.AME,
    cmb=components.CMB,
    dust=components.Dust,
    ff=components.FreeFree,
    radio=components.Radio,
    synch=components.Synchrotron,
)
    

def _download_BP_model(release: int, nside: int, cache: bool = True) -> Model:
    model_name = f'BP_test_model_r{release}_n{nside}.pkl'
    path_to_model = download_file(data_url + model_name, cache=cache)

    with open(path_to_model, 'rb') as f:
        return pickle.load(f)


def BP(release: int , nside: int, cache: bool = True) -> Model:
    """Loads the BeyondPlanck sky model for a given BP release. 

    The model is downloaded and cached using astropy.utils.

    Args:
    -----
    release (int):
        BeyondPlanck release number.
    nside (int):
        Healpix resolution parameter. Model is downloaded at the given nside.
    cache (bool):
        If True, the downloaded model is cached away for later used.

    Returns:
    (cosmoglobe.sky.Model):
        The BeyondPlanck sky model at a given nside for a release.
        
    """
    model = _download_BP_model(release, nside, cache=cache)
    return model
