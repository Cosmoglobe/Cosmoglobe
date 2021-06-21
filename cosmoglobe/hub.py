from .sky.model import Model

from astropy.utils.data import download_file
import pickle

data_url = 'http://cosmoglobe.uio.no/BeyondPlanck/precomputed/'

def load_model(path_to_model: str) -> Model:
    """Loads a model from file.
    
    Args:
    -----
    path_to_model (str):
        Path to the model file.

    Returns:
    --------
    model (cosmoglobe.sky.Model):
        Loaded cosmoglobe sky model.

    """
    with open(path_to_model, 'rb') as f:
        model =  pickle.load(f)

    if not isinstance(model, Model):
        raise TypeError(f'{path_to_model} is not a valid cosmoglobe.sky.Model')

    return model


def save_model(model: Model, filename: str) -> None:
    """Saves a model to file.
    
    Args:
    -----
    model (cosmoglobe.sky.Model):
        The sky model to save.
    filename (str):
        Save filename.

    """
    if not isinstance(model, Model):
        raise TypeError(f'{model} is not a valid cosmoglobe.sky.Model')

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


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