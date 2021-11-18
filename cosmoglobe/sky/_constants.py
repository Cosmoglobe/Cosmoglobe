from typing import Tuple
import astropy.constants as const
from astropy.units import Unit, Quantity

DEFAULT_BEAM_FWHM: Quantity = 0.0 * Unit("arcmin")
DEFAULT_OUTPUT_UNIT: Unit = Unit("uK_RJ")
DEFAULT_OUTPUT_UNIT_STR = "uK_RJ"
DEFAULT_GAL_CUT: Quantity = 10 * Unit("deg")
SIGNAL_UNITS: Tuple[Unit, Unit, Unit] = (Unit("K_RJ"), Unit("K_CMB"), Unit("Jy/sr"))

h: Quantity = const.h
c: Quantity = const.c
k_B: Quantity = const.k_B
T_0: Quantity = 2.7255 * Unit("K")
