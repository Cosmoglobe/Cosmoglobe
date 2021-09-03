import astropy.units as u
import numpy as np

import cosmoglobe.utils.constants as const


@u.quantity_input(freq=u.Hz, T=u.K)
def blackbody_emission(freq: u.Quantity, T: u.Quantity) -> u.Quantity:
    """Returns the blackbody emission.

    Computes the emission emitted by a blackbody with with temperature
    T at a frequency freq in SI units [W / m^2 Hz sr].

    Parameters
    ----------
    freq
        Frequency [Hz].
    T
        Temperature of the blackbody [K].

    Returns
    -------
    emission
        Blackbody emission [W / m^2 Hz sr].
    """

    # Avoiding overflow and underflow.
    T = T.astype(np.float64)

    term1 = (2 * const.h * freq ** 3) / const.c ** 2
    term2 = np.expm1((const.h * freq) / (const.k_B * T))

    emission = term1 / term2 / u.sr

    return emission.to("W / m^2 Hz sr")


@u.quantity_input(freq=u.Hz, T_e=u.K)
def gaunt_factor(freq: u.Quantity, T_e: u.Quantity) -> u.Quantity:
    """Returns the Gaunt factor.

    Computes the gaunt factor for a given frequency and electron
    temperaturein SI units.

    Parameters
    ----------
    freq
        Frequency [Hz].
    T_e
        Electron temperature [K].

    Returns
    -------
    gaunt_factor
        Gaunt Factor.
    """

    # Avoiding overflow and underflow.
    T_e = T_e.astype(np.float64)
    T_e = (T_e.to(u.kK)).value / 10
    freq = (freq.to(u.GHz)).value

    gaunt_factor = np.log(
        np.exp(5.96 - (np.sqrt(3) / np.pi) * np.log(freq * T_e ** -1.5)) + np.e
    )

    return u.Quantity(gaunt_factor, unit=u.dimensionless_unscaled)


@u.quantity_input(freq=u.Hz, T=u.K)
def thermodynamical_to_brightness(
    freq: u.Quantity, T: u.Quantity = const.T_0
) -> u.Quantity:
    """Conversion factor between K_CMB and K_RJ.

    Returns the conversion factor between thermodynamical and brightness
    temperatures [K_CMB -> K_RJ].

    Parmaeters
    ----------
    freq
        Frequency [K].

    Returns:
    --------
        Conversion factor.
    """

    x = (const.h * freq) / (const.k_B * T)

    return ((x ** 2 * np.exp(x)) / (np.expm1(x) ** 2)).si


@u.quantity_input(freq=u.Hz, T=u.K)
def brightness_to_thermodynamical(
    freq: u.Quantity, T: u.Quantity = const.T_0
) -> u.Quantity:
    """Conversion factor between K_RJ and K_CMB.

    Returns the conversion factor between thermodynamical and brightness
    temperatures [K_RJ -> K_CMB].

    Parmaeters
    ----------
    freq
        Frequency [K].

    Returns:
    --------
        Conversion factor.
    """

    return 1 / thermodynamical_to_brightness(freq, T)
