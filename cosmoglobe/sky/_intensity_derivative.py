from typing import Protocol, Dict

from astropy.units import Quantity, Unit
import numpy as np

import cosmoglobe.sky._constants as const


class IntensityDerivative(Protocol):
    """Interface for an intensity derivate."""

    def __call__(self, freqs: Quantity) -> Quantity:
        """Returns the intensity derivative given some frequencies.

        Parameters
        ----------
        freqs
            Frequencies for which to compute the intensity derivative.

        Returns
        -------
            The intensity derivative given input and output units.
        """


class RayleighJeansIntensityDerivative:
    """Intensity derivative for K_RJ."""

    def __call__(self, freqs: Quantity) -> Quantity:
        factor = (2 * const.k_B * freqs ** 2) / const.c ** 2
        # We specify that that the kelvin in this expression refers to K_CMB
        # and divide by sr to make this quantity compatible with the emission
        # amplitudes
        factor *= Unit("K") / Unit("K_RJ")
        factor /= Unit("sr")

        return factor


class CMBIntensityDerivative:
    """Intensity derivative for K_CMB."""

    def __call__(self, freqs: Quantity) -> Quantity:
        x = (const.h * freqs) / (const.k_B * const.T_0)
        term1 = (2 * const.h * freqs ** 3) / (const.c ** 2 * np.expm1(x))
        term2 = np.exp(x) / np.expm1(x)
        term3 = (const.h * freqs) / (const.k_B * const.T_0 ** 2)

        factor = term1 * term2 * term3
        # We specify that that the kelvin in this expression refers to K_CMB
        # and divide by sr to make this quantity compatible with the emission
        # amplitudes
        factor *= Unit("K") / Unit("K_CMB")
        factor /= Unit("sr")
        return factor


class IRASIntensityDerivative:
    """Intensity derivative for MJY/sr."""

    def __call__(self, freqs: Quantity) -> Quantity:
        return np.mean(freqs) / freqs


INTENSITY_DERIVATIVES: Dict[Unit, IntensityDerivative] = {
    Unit("K_RJ"): RayleighJeansIntensityDerivative(),
    Unit("K_CMB"): CMBIntensityDerivative(),
    Unit("Jy/sr"): IRASIntensityDerivative(),
}


def get_intensity_derivative(unit: Unit) -> IntensityDerivative:
    """Returns the related intensity derivative for a given unit."""

    for intensity_derivative in INTENSITY_DERIVATIVES:
        if intensity_derivative.is_equivalent(unit):
            return INTENSITY_DERIVATIVES[intensity_derivative]
    else:
        raise KeyError("unit does not match any known intensity derivatives.")
