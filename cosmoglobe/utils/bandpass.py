from typing import Dict, Iterable, TYPE_CHECKING

import astropy.units as u
import numpy as np
from scipy.interpolate import RectBivariateSpline

import cosmoglobe.utils.constants as const

if TYPE_CHECKING:
    from cosmoglobe.sky.base import SkyComponent


@u.quantity_input(bandpass=(u.Jy / u.sr, u.K), freqs=u.Hz)
def get_normalized_bandpass(
    bandpass: u.Quantity, freqs: u.Quantity, unit: u.UnitBase = u.K
) -> u.Quantity:
    """Returns a bandpass profile normalized to unity when integrated over all
    frequencies corresponding to the profile in units of K_RJ.

    Parameters
    ----------
    bandpass
        Bandpass profile in signal units.
    freqs
        Frequencies corresponding to the bandpass profile.
    unit
        The unit we want to convert to.

    Returns
    -------
        Normalized bandpass
    """

    bandpass = bandpass.to(unit=unit, equivalencies=u.brightness_temperature(freqs))

    return bandpass / np.trapz(bandpass, freqs)


@u.quantity_input(bandpass=u.Hz ** -1, freqs=u.Hz)
def get_bandpass_coefficient(
    bandpass: u.Quantity, freqs: u.Quantity, output_unit: u.UnitBase
) -> u.Quantity:
    """Returns the bandpass coefficient.

    Computes the bandpass coefficent for a map that has been calibrated in
    K_RJ to some output unit, after taking into account the bandpass.

    Parameters
    ----------
    bandpass
        Normalized bandpass profile [Hz^-1].
    freqs
        Frequencies corresponding to the bandpass profile.
    unit
        The unit we want to convert to.

    Returns
    -------
    bandpass_coefficient
        The bandpass coefficient.
    """

    intensity_derivative_i = b_rj(freqs)

    # Selecting the intensity derivative expression given an output_unit
    try:
        output_unit = u.Unit(output_unit)
        if u.K in output_unit.si.bases:
            intensity_derivative_j = b_rj(freqs)
        elif output_unit == u.MJy / u.sr:
            intensity_derivative_j = b_iras(freqs, np.mean(freqs))

    except ValueError:
        if isinstance(output_unit, str):
            if output_unit.lower().endswith("k_rj"):
                intensity_derivative_j = b_rj(freqs)
            elif output_unit.lower().endswith("k_cmb"):
                intensity_derivative_j = b_cmb(freqs)
            else:
                raise NotImplementedError(
                    f"output_unit {output_unit} is not implemented"
                )
    bandpass_coefficient = np.trapz(
        bandpass * intensity_derivative_i, freqs
    ) / np.trapz(bandpass * intensity_derivative_j, freqs)

    return bandpass_coefficient


def get_bandpass_scaling(
    freqs: u.Quantity, bandpass: u.Quantity, comp: "SkyComponent"
) -> np.ndarray:
    """Returns the frequency scaling factor given a bandpass profile and a
    corresponding frequency array.

    Parameters
    ----------
    freqs
        List of frequencies.
    bandpass
        Normalized bandpass profile.
    Returns
    -------
    bandpass_scaling
        Frequency scaling factor given a bandpass.
    """

    grid = get_interpolation_grid(comp.spectral_parameters)
    if not grid:
        # Component does not have any spatially varying spectral parameters.
        # In this case we simply integrate the emission at each frequency
        # weighted by the bandpass.
        freq_scaling = comp._get_freq_scaling(
            freqs, comp.freq_ref, **comp.spectral_parameters
        )
        integral = np.trapz(freq_scaling * bandpass, freqs)
        if np.ndim(integral) > 0:
            return np.expand_dims(integral, axis=1)
        return integral
    elif len(grid) == 1:
        # Component has one sptatially varying spectral parameter. In this
        # scenario we perform a 1D-interpolation in spectral parameter space.
        return interp1d(freqs, bandpass, grid, comp)
    elif len(grid) == 2:
        # Component has two sptatially varying spectral parameter. In this
        # scenario we perform a 2D-interpolation in spectral parameter space.
        return interp2d(freqs, bandpass, grid, comp)
    else:
        raise NotImplementedError(
            "Bandpass integration for comps with more than two spectral "
            "parameters is not currently supported"
        )


def get_interpolation_grid(
    spectral_parameters: Dict[str, u.Quantity]
) -> Dict[str, u.Quantity]:
    """Returns a interpolation range.

    Computes the interpolation range of the spectral parameters of a
    sky component. We use a regular grid with n points for the range.

    Parameters
    ----------
    spectral_parameters
        Dictionary containing the spectral parameters of a given component.

    Returns
    -------
    interp_parameters
        Dictionary with a interpolation grid for each spatially varying
        spectral parameter.
    """

    dim = 0
    for spectral_parameter in spectral_parameters.values():
        if spectral_parameter.size > 3:
            dim += 1

    grid = {}
    if dim == 0:
        return grid
    elif dim == 1:
        n = 1000
    elif dim == 2:
        n = 100

    for key, value in spectral_parameters.items():
        if value.size > 3:
            grid_range = np.linspace(np.amin(value), np.amax(value), n)
            grid[key] = grid_range

    return grid


def interp1d(
    freqs: u.Quantity, bandpass: u.Quantity, grid: u.Quantity, comp: "SkyComponent"
) -> Iterable[float]:
    """Performs 1D bandpass integration.

    This function uses numpy interp1d to perform bandpass integration
    for a sky component and returns the frequency scaling factor.

    Parameters
    ----------
    freqs
        Frequencies corresponding to the bandpass profile.
    bandpass:
        Normalized bandpass profile in units of 1/Hz.
    grid
        Linearly spaced grid points of the spectral parameters for which to
        integrate over.
    comp
        A Cosmoglobe sky component object.

    Returns
    -------
    scaling
        Frequency scaling factor obtained by integrating over the bandpass.
    """

    # Grid dictionary only has one item
    ((key, grid),) = grid.items()

    if comp._is_polarized:
        integrals = np.zeros((len(grid), 3))
    else:
        integrals = np.zeros((len(grid), 1))

    for idx, grid_point in enumerate(grid):
        freq_scaling = comp._get_freq_scaling(freqs, comp.freq_ref, **{key: grid_point})
        integrals[idx] = np.trapz(freq_scaling * bandpass, freqs)

    # We transpose the array to make it into row format similar to how
    # regular IQU maps are stored
    integrals = np.transpose(integrals)

    scaling = [
        np.interp(comp.spectral_parameters[key][idx].value, grid.value, col)
        for idx, col in enumerate(integrals)
    ]

    return scaling


def interp2d(
    freqs: u.Quantity, bandpass: u.Quantity, grid: u.Quantity, comp: "SkyComponent"
) -> u.Quantity:
    """Performs 1D bandpass integration.

    This function uses scipy's RectBivariateSpline to perform  bandpass
    integration for a sky component and returns the frequency scaling factor.

    TODO: Test this feature for a production chain.

    Parameters
    ----------
    freqs
        Frequencies corresponding to the bandpass profile.
    bandpass:
        Normalized bandpass profile in units of 1/Hz.
    grid
        Linearly spaced 2D grid of the spectral parameters for which to
        integrate over.
    comp
        A Cosmoglobe sky component object.

    Returns
    -------
    scaling
        Frequency scaling factor obtained by integrating over the bandpass.
    """

    # Make n x n mesh grid for the spectral parameters
    n = len(list(grid.values())[0])
    if comp._is_polarized:
        integrals = np.zeros((n, n, 3))
    else:
        integrals = np.zeros((n, n, 1))

    mesh_grid = {
        key: value for key, value in zip(grid.keys(), np.meshgrid(*grid.values()))
    }

    for i in range(n):
        for j in range(n):
            grid_spectrals = {key: value[i, j] for key, value in mesh_grid.items()}
            freq_scaling = comp._get_freq_scaling(
                freqs, comp.freq_ref, **grid_spectrals
            )
            integrals[i, j] = np.trapz(freq_scaling * bandpass, freqs)
    integrals = np.transpose(integrals)

    scaling = []
    for idx, col in enumerate(integrals):
        f = RectBivariateSpline(*grid.values(), col)
        packed_spectrals = [
            spectral[idx] if spectral.shape[0] == 3 else spectral[0]
            for spectral in comp.spectral_parameters.values()
        ]
        scaling.append(f(*packed_spectrals, grid=False))

    return scaling


@u.quantity_input(freq=u.Hz)
def b_rj(freq: u.Quantity) -> u.Quantity:
    """Returns the intensity derivative for K_RJ.

    Computes the intensity derivate factor [Jy/sr -> K_RJ].

    Parameters
    ----------
    freq
        Frequency [Hz].

    Returns
    -------
        Intensity derivative factor.
    """

    return (2 * const.k_B * freq ** 2) / (const.c ** 2 * u.sr)


@u.quantity_input(freq=u.Hz, T=u.K)
def b_cmb(freq: u.Quantity, T: u.Quantity = const.T_0) -> u.Quantity:
    """The intensity derivative for K_CMB.

    Computes the intensity derivate factor [Jy/sr -> K_CMB].

    Parameters
    ----------
    freq
        Frequency [Hz].
    T
        The CMB emperature [K]. Default is const.T_0.

    Returns
    -------
        Bandpass coefficient.
    """

    x = (const.h * freq) / (const.k_B * T)

    term1 = (2 * const.h * freq ** 3) / (const.c ** 2 * np.expm1(x))
    term2 = np.exp(x) / np.expm1(x)
    term3 = (const.h * freq) / (const.k_B * T ** 2)

    return term1 * term2 * term3 / u.sr


@u.quantity_input(freq=u.Hz, freq_ref=u.Hz)
def b_iras(freq: u.Quantity, freq_ref: u.Quantity) -> u.Quantity:
    """The intensity derivative in the IRAS unit convention (MJy/sr).

    Parameters
    ----------
    freq
        Frequency [Hz].
    freq_ref
        Reference frequency [Hz].

    Returns
    -------
        Bandpass coefficient.
    """

    return freq_ref / freq
