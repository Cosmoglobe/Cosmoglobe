from typing import Protocol, Dict

from astropy.units import Quantity, Unit
import numpy as np

import cosmoglobe.sky._constants as const


class IntensityDerivative(Protocol):
    """Protocol defining the interface for an intensity derivate."""

    def __call__(self, freqs: Quantity) -> Quantity:
        """Returns the intensity derivative given some frequencies.

        Parameters
        ----------
        freqs
            Frequencies for which to compute the intensity derivative.

        Returns
        -------
            The intensity derivative unit conversion factor.
        """


def bnu_prime_RJ(freqs: Quantity) -> Quantity:
    """Intensitity derivative for K_RJ (dB_nu(T)_dT in the Rayleigh-Jeans limit)."""

    factor = (2 * const.k_B * freqs ** 2) / const.c ** 2
    factor *= Unit("K") / (Unit("K_RJ") * Unit("sr"))

    return factor


def bnu_prime_CMB(freqs: Quantity) -> Quantity:
    """Intensity derivative for K_CMB (dB_nu(T)/dT)."""

    x = (const.h * freqs) / (const.k_B * const.T_0)
    A = 2 * const.k_B * freqs ** 2 / const.c ** 2
    factor = A * x ** 2 * np.exp(x) / np.expm1(x) ** 2
    factor *= Unit("K") / (Unit("K_CMB") * Unit("sr"))

    return factor


def dI_nu_dI_c(freqs: Quantity) -> Quantity:
    """Intensity derivative for MJy/sr (IRAS convention)."""

    return np.mean(freqs) / freqs


INTENSITY_DERIVATIVE_MAPPINGS: Dict[Unit, IntensityDerivative] = {
    Unit("K_RJ"): bnu_prime_RJ,
    Unit("K_CMB"): bnu_prime_CMB,
    Unit("Jy/sr"): dI_nu_dI_c,
}


def get_intensity_derivative(unit: Unit) -> IntensityDerivative:
    """Returns the intensity derivative related to a unit."""

    for intensity_derivative in INTENSITY_DERIVATIVE_MAPPINGS:
        if intensity_derivative.is_equivalent(unit):
            return INTENSITY_DERIVATIVE_MAPPINGS[intensity_derivative]

    raise KeyError("unit does not match any known intensity derivatives.")
