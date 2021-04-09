import numpy as np
import astropy.units as u


def _extract_scalars(spectral_parameters):
    """Extracts scalars from a spectral_parameter dict. If a map is constant
    in all dims, then it is converted to a (ndim,) array"""
    scalars = {}
    spectral_parameters['test'] = [5,3,3]*u.GHz
    for key, value in spectral_parameters.items():
        if value.ndim > 1:
            uniques = [np.unique(col) for col in value]
            all_is_unique = all([len(col) == 1 for col in uniques])
            if all_is_unique:
                scalars[key] = np.concatenate(uniques)

        else:
            print(key)
            scalars[key] = value

    return scalars

def _set_spectral_units(maps):
    """    
    TODO: Figure out how to correctly detect unit of spectral map in chain.
          Until then, a hardcoded dict is used:
    """
    units = {
        'T': u.K,
        'Te': u.K,
        'nu_p' : u.GHz,
    }
    for map_ in maps:
        if map_ in units:
            maps[map_] *= units[map_]
        else:
            maps[map_] *= u.dimensionless_unscaled

    return maps