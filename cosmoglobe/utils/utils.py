from typing import Dict, Iterable

import astropy.units as u
import numpy as np

import cosmoglobe.utils.functions as F


def extract_scalars(spectral_parameters: Dict[str, u.Quantity]):
    """Extracts scalars from a spectral_parameter dict.

    If a map is constant in all dims, then it is converted to a (ndim,)
    array.
    """

    scalars = {}
    for key, value in spectral_parameters.items():
        if not isinstance(value, u.Quantity):
            value = u.Quantity(value, unit=u.dimensionless_unscaled)
        if value.size > 1:
            if value.ndim > 1:
                uniques = [np.unique(col) for col in value.value]
                all_is_unique = all([len(col) == 1 for col in uniques])
            else:
                uniques = np.unique(value.value)
                all_is_unique = len(uniques) == 1
            if all_is_unique:
                scalars[key] = uniques * value.unit
        else:
            scalars[key] = value

    return scalars


def set_spectral_units(maps: Iterable[u.Quantity]) -> u.Quantity:
    """Sets the spectral value for a specific spectral parameter name.

    Parameters
    ----------
    maps
        List of maps.

    Returns
    -------
    maps
        List of maps with new units.
    """

    # TODO: Figure out how to correctly detect unit of spectral map in chain.
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


def str_to_astropy_unit(unit: str) -> u.UnitBase:
    """Converts a K_RJ or a K_CMB string to `u.K` unit.

    This function preserves the prefix.

    Parameters
    ----------
    unit
        String to convert to a astropy unit.

    Returns
    -------
    output_unit
        Astropy unit.
    """

    try:
        output_unit = u.Unit(unit)
    except ValueError:
        if unit.lower().endswith("k_rj"):
            output_unit = u.Unit(unit[:-3])

        elif unit.lower().endswith("k_cmb"):
            output_unit = u.Unit(unit[:-4])

    return output_unit


def to_unit(emission: u.Quantity, freqs: u.Quantity, unit: u.UnitBase) -> u.Quantity:
    """Converts the unit of the emission.

    Parameters
    ----------
    emission
        Emission whos unit we want to convert.
    freqs
        Frequency or a list of frequencies for which the emission was
        obtained.
    unit
        Unit to convert to.

    Returns
    -------
    emission
        Emission with new units.
    """

    try:
        unit = u.Unit(unit)
        emission = emission.to(unit, equivalencies=u.brightness_temperature(freqs))

    except ValueError:
        if unit.lower().endswith("k_rj"):
            unit = u.Unit(unit[:-3])
        elif unit.lower().endswith("k_cmb"):
            unit = u.Unit(unit[:-4])
            emission *= F.brightness_to_thermodynamical(freqs)
        emission.to(unit)

    return emission


def gaussian_beam_2D(r: np.ndarray, sigma: float) -> np.ndarray:
    """Returns the Gaussian beam in 2D in polar coordinates.

    Parameters
    ----------
    r
        Angular distance.
    sigma
        The sigma of the Gaussian (beam radius).

    Returns
    -------
        Gaussian beam.
    """

    return r * np.exp(-(r ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
