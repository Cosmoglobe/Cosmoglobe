import astropy.constants as const
from astropy.units import Unit

DEFAULT_BEAM_FWHM = 0.0 * Unit("arcmin")
DEFAULT_OUTPUT_UNIT = Unit("uK_RJ")
DEFAULT_OUTPUT_UNIT_STR = "uK_RJ"
DEFAULT_GAL_CUT = 10 * Unit("deg")
SIGNAL_UNITS = (Unit("K_RJ"), Unit("K_CMB"), Unit("Jy/sr"))

h = const.h
c = const.c
k_B = const.k_B
T_0 = 2.7255 * Unit("K")
