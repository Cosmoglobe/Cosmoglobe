from cosmoglobe.utils.functions import brightness_to_thermodynamical

import numpy as np
import astropy.units as u
class NsideError(Exception):
    """Raised if there is a NSIDE related problem"""
    pass

class ModelError(Exception):
    """Raised if there is a Model related problem"""
    pass


def _extract_scalars(spectral_parameters):
    """Extracts scalars from a spectral_parameter dict. If a map is constant
    in all dims, then it is converted to a (ndim,) array"""
    scalars = {}
    for key, value in spectral_parameters.items():
        if value.size > 1:
            uniques = [np.unique(col) for col in value.value]
            all_is_unique = all([len(col) == 1 for col in uniques])
            if all_is_unique:
                scalars[key] = uniques*value.unit

        else:
            scalars[key] = value
    return scalars


def _set_spectral_units(maps):
    #TODO: Figure out how to correctly detect unit of spectral map in chain.
    #      Until then, a hardcoded dict is used:
    units = dict(
        T=u.K,
        Te=u.K,
        nu_p=u.GHz,
    )
    for map_ in maps:
        if map_ in units:
            maps[map_] *= units[map_]
        else:
            maps[map_] *= u.dimensionless_unscaled

    return maps


def _str_to_astropy_unit(unit):
    try:
        output_unit = u.Unit(unit)
    except ValueError:
        if unit.lower().endswith('k_rj'):
            output_unit = u.Unit(unit[:-3])
            
        elif unit.lower().endswith('k_cmb'):
            output_unit = u.Unit(unit[:-4])

    return output_unit


def emission_to_unit(emission, freqs, unit):
    try:
        unit = u.Unit(unit)
        emission = emission.to(
            unit, equivalencies=u.brightness_temperature(freqs)
        )

    except ValueError:
        if unit.lower().endswith('k_rj'):
            unit = u.Unit(unit[:-3])
        elif unit.lower().endswith('k_cmb'):
            unit = u.Unit(unit[:-4])  
            print("here")
            emission *= brightness_to_thermodynamical(freqs)
        emission.to(unit)

    return emission


def gaussian_beam_2D(r, sigma):
    """Returns the Gaussian beam in 2D in polar coordinates.

    Parameters:
    -----------
    r : float
        Angular distance.
    sigma : float
        The sigma of the Gaussian (beam radius).
    
    """
    return r*np.exp(-(r**2)/(2 * sigma**2))/(sigma*np.sqrt(2*np.pi))
