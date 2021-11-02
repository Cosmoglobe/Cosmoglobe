import astropy.units as u
import numpy as np

import cosmoglobe.sky._constants as const



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
