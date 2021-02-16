from dataclasses import dataclass
import astropy.units as u 
import json

@dataclass
class ModelParameters:
    """
    Data class for the parameters of a model.

    """
    type: str
    polarization: bool
    unit: str
    nu_ref: float
    nside: int
    fwhm: float

    def __str__(self):
        return str(self.__dict__)


def unpack_config(config_file):
    """
    Loads the configuration file from json.

    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def get_params_from_config(config_file):
    """
    Returns a dictionary containing parameters from the configuration file.

    """
    params = {}
    for comp in config_file:
        params[comp] = ModelParameters(
            type=config_file[comp]['type'],
            polarization=config_file[comp]['polarization'],
            unit=config_file[comp]['unit'],
            nu_ref=config_file[comp]['nu_ref']*u.GHz,
            nside=config_file[comp]['nside'],
            fwhm=config_file[comp]['fwhm']*u.arcmin,
        )
    
    return params