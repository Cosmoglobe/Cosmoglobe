from typing import Callable, Dict
from astropy.units import Quantity, Unit
import numpy as np

import cosmoglobe.sky._constants as const


def b_rj(freq: Quantity) -> Quantity:
    """Returns the intensity derivative for K_RJ."""

    return (2 * const.k_B * freq ** 2) / (const.c ** 2 * Unit("sr"))


def b_cmb(freq: Quantity, T: Quantity = const.T_0) -> Quantity:
    """The intensity derivative for K_CMB."""

    x = (const.h * freq) / (const.k_B * T)
    term1 = (2 * const.h * freq ** 3) / (const.c ** 2 * np.expm1(x))
    term2 = np.exp(x) / np.expm1(x)
    term3 = (const.h * freq) / (const.k_B * T ** 2)

    return term1 * term2 * term3 / Unit("sr")


def b_iras(freq: Quantity) -> Quantity:
    """The intensity derivative in the IRAS unit convention (MJy/sr)."""

    return np.mean(freq) / freq


INTENSITY_DERIVATIVE_MAPPINGS: Dict[str, Callable[[Quantity], Quantity]] = {
    "K_RJ": b_rj,
    "K_CMB": b_cmb,
    "Jy/sr": b_iras,
}
