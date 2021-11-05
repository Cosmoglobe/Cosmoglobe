from typing import Protocol, Dict, Union

from astropy.units import Quantity, Unit, UnitBase
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
        return (2 * const.k_B * freqs ** 2) / (const.c ** 2 * Unit("sr"))


class CMBIntensityDerivative:
    """Intensity derivative for K_CMB."""

    def __call__(self, freqs: Quantity) -> Quantity:
        x = (const.h * freqs) / (const.k_B * const.T_0)
        term1 = (2 * const.h * freqs ** 3) / (const.c ** 2 * np.expm1(x))
        term2 = np.exp(x) / np.expm1(x)
        term3 = (const.h * freqs) / (const.k_B * const.T_0 ** 2)

        return term1 * term2 * term3 / Unit("sr")


class IRASIntensityDerivative:
    """Intensity derivative for MJY/sr."""

    def __call__(self, freqs: Quantity) -> Quantity:
        return np.mean(freqs) / freqs


INTENSITY_DERIVATIVES: Dict[str, IntensityDerivative] = {
    "rj": RayleighJeansIntensityDerivative(),
    "cmb": CMBIntensityDerivative(),
    "jy": IRASIntensityDerivative(),
}


def get_intensity_derivative(
    unit: Union[str, UnitBase]
) -> IntensityDerivative:
    """Returns the related intensity derivative for a given unit."""

    if isinstance(unit, str):
        unit = unit.lower()
        if unit in ("k", "uk", "mk"):
            unit = "rj"
    else:
        if unit.is_equivalent(Unit("K")):
            unit = "rj"
        else:
            unit = unit.to_string().replace(" ", "").lower()

    for intensity_derivative in INTENSITY_DERIVATIVES:
        if intensity_derivative in unit:
            return INTENSITY_DERIVATIVES[intensity_derivative]
    else:
        raise KeyError("unit does not match any known intensity derivatives.")
