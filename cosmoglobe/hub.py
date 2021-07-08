from cosmoglobe.sky.model import Model
from cosmoglobe.sky import components
from cosmoglobe.utils.utils import ModelError

from astropy.utils.data import download_file
from astropy.io.misc import fnpickle, fnunpickle
import numpy as np

data_url = 'http://cosmoglobe.uio.no/BeyondPlanck/precomputed/'

COSMOGLOBE_COMPS = dict(
    dust=components.Dust,
    synch=components.Synchrotron,
    ff=components.FreeFree,
    ame=components.AME,
    cmb=components.CMB,
    radio=components.Radio
)


def save_model(model, filename):
    """Saves a model to file in form of a dictionary.
    
    Args:
    -----
    model (cosmoglobe.sky.Model):
        The sky model to save.
    filename (str):
        Save filename.

    """

    model_dict = {}
    try:
        for comp in model:
            freq_ref =  comp.freq_ref
            if freq_ref is not None:
                if freq_ref.ndim > 0:
                    # freq_ref is stored as (3,1) or (1,1) arrays in model for broadcasting
                    freq_ref = np.squeeze(freq_ref)
                    freq_ref = [freq_ref[0].value, freq_ref[-1].value]*freq_ref.unit
                else:
                    freq_ref = freq_ref
            
            comp_dict = {
                comp.name : {
                    'amp': comp.amp,
                    'freq_ref': freq_ref,
                    'spectral_parameters': comp.spectral_parameters,
                }
            }
            model_dict.update(comp_dict)

    except AttributeError:
        raise ModelError(
            'Model is not compatible with the current Cosmoglobe Sky Model'
        )

    fnpickle(model_dict, filename)


def load_model(path_to_model):
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
    model_dict =  fnunpickle(path_to_model)

    model = model_from_dict(model_dict)

    return model



def model_from_dict(model_dict):
    """Creates and returns a cosmoglobe.sky.Model from a model dictionary."""
    model = Model()
    for key, value in model_dict.items():
        try:
            comp = COSMOGLOBE_COMPS[key]
        except KeyError:
            raise ModelError(
                'Model components does not match the Cosmoglobe Sky Model'
            )

        try:
            amp = value['amp']
            freq_ref = value['freq_ref']
            spectral_parameters = value['spectral_parameters']
        except KeyError as e:
            raise e

        if freq_ref is not None: #CMB is initialized without a reference freq
            model.insert(comp(key, amp, freq_ref, **spectral_parameters))    
        else:
            model.insert(comp(key, amp, **spectral_parameters))    
        
    return model

    

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
