import astropy.units as u
from astropy.units.equivalencies import spectral
from astropy.units.quantity import Quantity
import numpy as np

from cosmoglobe.refactored_sky.enums import SkyComponentLabel, SkyComponentType
from cosmoglobe.refactored_sky.SEDs import (
    GauntFactor,
    PowerLaw,
    ModifiedBlackBody,
    SPDust2,
    ThermodynamicToBrightness,
)
from cosmoglobe.data import DATA_DIR

RADIO_CATALOG = DATA_DIR / "radio_catalog.dat"


class Synchrotron:
    """Synchrotron sky component."""

    label = SkyComponentLabel.SYNCH
    component_type = SkyComponentType.DIFFUSE
    SED = PowerLaw()

    def __init__(self, amp: u.uK, freq_ref: u.GHz, beta: Quantity):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = {"beta": beta}


class ThermalDust:
    """Thermal dust sky component."""

    label = SkyComponentLabel.DUST
    component_type = SkyComponentType.DIFFUSE
    SED = ModifiedBlackBody()

    def __init__(self, amp: u.uK, freq_ref: u.GHz, beta: Quantity, T: u.K):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = {"beta": beta, "T": T}



class CosmicMicrowaveBackground:
    """Thermal dust sky component."""

    label = SkyComponentLabel.CMB
    component_type = SkyComponentType.DIFFUSE
    SED = ThermodynamicToBrightness()

    def __init__(self, amp: u.uK):
        self.amp = amp
        self.freq_ref = 1 * u.GHz
        self.spectral_parameters = {}


class FreeFree:
    """Thermal dust sky component."""

    label = SkyComponentLabel.FF
    component_type = SkyComponentType.DIFFUSE
    SED = GauntFactor()

    def __init__(self, amp: u.uK, freq_ref: u.GHz, T_e: u.K):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = {"T_e": T_e}


class SpinningDust:
    """Thermal dust sky component."""

    label = SkyComponentLabel.AME
    component_type = SkyComponentType.DIFFUSE
    SED = SPDust2()

    def __init__(self, amp: u.uK, freq_ref: u.GHz, freq_peak: u.GHz):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = {"freq_peak": freq_peak}

class RadioSources:
    """Radio sources sky component."""

    label = SkyComponentLabel.RADIO
    component_type = SkyComponentType.POINTSOURCE
    SED = PowerLaw()
    catalog = np.loadtxt(RADIO_CATALOG, usecols=(0, 1)).transpose()

    def __init__(self, amp: u.uK, freq_ref: u.GHz, alpha: Quantity):
        self.amp = amp
        self.freq_ref = freq_ref
        self.spectral_parameters = {"alpha": alpha}
