"""Here we define K_CMB and K_RJ as custom astropy units. Unforunately, these 
unit conventions are not built into astropy, meaning that we need to define
them our selves. In the following code, we follow the implementation in pysm3

Reference: https://github.com/galsci/pysm/blob/main/pysm3/__init__.py
"""

from typing import Any, List

from astropy.units import (
    add_enabled_units,
    brightness_temperature,
    def_unit,
    quantity_input,
    Quantity,
    thermodynamic_temperature,
    Unit,
)


@quantity_input(freqs="Hz")
def cmb_equivalencies(freqs: Quantity) -> List[Any]:
    """Custom equivalency representing the conversion between Kelvin Rayleigh-Jeans and CMB units.

    Parameters
    ----------
    freqs
        Frequencies at which to perform the conversion.
    """

    [(_, _, Jy_to_CMB, CMB_to_Jy)] = thermodynamic_temperature(freqs)
    [(_, _, Jy_to_RJ, RJ_to_Jy)] = brightness_temperature(freqs)

    def RJ_to_CMB(T_RJ):
        return Jy_to_CMB(RJ_to_Jy(T_RJ))

    def CMB_to_RJ(T_CMB):
        return Jy_to_RJ(CMB_to_Jy(T_CMB))

    return [
        (Unit("K_RJ"), Unit("K_CMB"), RJ_to_CMB, CMB_to_RJ),
        (Unit("Jy") / Unit("sr"), Unit("K_RJ"), Jy_to_RJ, RJ_to_Jy),
        (Unit("Jy") / Unit("sr"), Unit("K_CMB"), Jy_to_CMB, CMB_to_Jy),
    ]


def_unit(
    "K_CMB",
    doc="Kelvin CMB: Thermodynamic temperature units.",
    format={"generic": "K_CMB", "latex": "K_{{CMB}}"},
    prefixes=True,
    namespace=globals(),
)

def_unit(
    "K_RJ",
    prefixes=True,
    format={"generic": "K_RJ", "latex": "K_{{RJ}}"},
    doc="Kelvin Rayleigh-Jeans: Brightness temperature.",
    namespace=globals(),
)

# Manually adding prefixes to units
add_enabled_units([uK_RJ, mK_RJ, K_RJ, kK_RJ, MK_RJ])
add_enabled_units([uK_CMB, mK_CMB, K_CMB, kK_CMB, MK_CMB])
