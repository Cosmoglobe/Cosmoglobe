import astropy.units as u
from astropy.units import Quantity
import numpy as np

from cosmoglobe.data import DATA_DIR
from cosmoglobe.refactored_sky.func import blackbody_emission, gaunt_factor
import cosmoglobe.refactored_sky.constants as const

SPDUST2_FILE = DATA_DIR / "spdust2_cnm.dat"


class PowerLaw:
    """SED for Synchrotron and Radio emission."""

    def get_freq_scaling(
        self,
        freqs: u.GHz,
        freq_ref: u.GHz,
        index: Quantity,
    ) -> Quantity:
        """See base class."""

        return (freqs / freq_ref) ** index


class ModifiedBlackBody:
    """SED for Thermal dust emission."""

    def get_freq_scaling(
        self,
        freqs: u.GHz,
        freq_ref: u.GHz,
        beta: Quantity,
        T: u.K,
    ) -> Quantity:
        """See base class."""

        blackbody_ratio = blackbody_emission(freqs, T) / blackbody_emission(freq_ref, T)
        scaling = (freqs / freq_ref) ** (beta - 2) * blackbody_ratio

        return scaling


class ThermodynamicToBrightness:
    """SED for the Cosmic Microwave Background emission."""

    def get_freq_scaling(
        self,
        freqs: u.GHz,
        T: u.K = const.T_0,
    ) -> Quantity:
        """See base class."""

        x = (const.h * freqs) / (const.k_B * T)
        scaling_factor = ((x ** 2 * np.exp(x)) / (np.expm1(x) ** 2)).si

        return np.expand_dims(scaling_factor(freqs), axis=0)


class GauntFactor:
    """SED for Free-free emission."""

    def get_freq_scaling(self, freqs: u.GHz, freq_ref: u.GHz, T_e: u.K) -> Quantity:
        """See base class."""

        gaunt_factor_ratio = gaunt_factor(freqs, T_e) / gaunt_factor(freq_ref, T_e)
        scaling = (freq_ref / freqs) ** 2 * gaunt_factor_ratio

        return scaling


class SPDust2:
    """SED for spinning dust emission."""

    SPINNING_DUST_TEMPLATE = np.loadtxt(SPDUST2_FILE).transpose()
    template_freq = Quantity(SPINNING_DUST_TEMPLATE[0], unit="GHz")
    template_amp = Quantity(SPINNING_DUST_TEMPLATE[1], unit="Jy/sr")
    template_amp = template_amp.to(
        "uK", equivalencies=u.brightness_temperature(template_freq)
    )

    def get_freq_scaling(
        self,
        freqs: u.GHz,
        freq_ref: u.GHz,
        freq_peak: u.GHz,
    ) -> Quantity:
        """See base class."""

        peak_scale = 30 * u.GHz / freq_peak

        interp = np.interp(
            (freqs * peak_scale).si.value,
            self.template_freq.si.value,
            self.template_amp.si.value,
            left=0.0,
            right=0.0,
        )
        interp_ref = np.interp(
            (freq_ref * peak_scale).si.value,
            self.template_freq.si.value,
            self.template_amp.si.value,
        )

        scaling = interp / interp_ref
        return scaling

