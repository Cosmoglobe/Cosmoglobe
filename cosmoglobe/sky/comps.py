from pathlib import Path

from astropy.units import Quantity, Unit, brightness_temperature
import numpy as np

from cosmoglobe.sky.basecomponent import DiffuseComponent, PointSourceComponent
from cosmoglobe.utils.functions import blackbody_emission

DATA_DIR = Path(__file__).parent.parent.resolve() / "data"
RADIO_CATALOG = DATA_DIR / "radio_catalog.dat"
SPDUST2_FILE = DATA_DIR / "spdust2_cnm.dat"


class Synchrotron(DiffuseComponent):
    """Synchrotron component."""

    label = "synch"

    def __init__(self, amp: Quantity, freq_ref: Quantity, beta: Quantity) -> None:
        """Initializing base classes."""

        super().__init__(self.label, amp, freq_ref, beta=beta)

    def _get_freq_scaling(self, freqs: Quantity, beta: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        return (freqs / self.freq_ref) ** beta


class Dust(DiffuseComponent):
    """Dust component."""

    label = "dust"

    def __init__(
        self, amp: Quantity, freq_ref: Quantity, beta: Quantity, T: Quantity
    ) -> None:
        """Initializing base classes."""

        super().__init__(self.label, amp, freq_ref, beta=beta, T=T)

    def _get_freq_scaling(  # type: ignore
        self, freqs: Quantity, beta: Quantity, T: Quantity
    ) -> Quantity:

        """See base class."""

        blackbody_ratio = blackbody_emission(freqs, T) / blackbody_emission(
            self.freq_ref, T
        )
        scaling = (freqs / self.freq_ref) ** (beta - 2) * blackbody_ratio

        return scaling


class AME(DiffuseComponent):
    """AME component."""

    label = "ame"

    SPINNING_DUST_TEMPLATE = np.loadtxt(SPDUST2_FILE).transpose()

    def __init__(self, amp: Quantity, freq_ref: Quantity, freq_peak: Quantity) -> None:
        """Initializing base classes."""

        super().__init__(self.label, amp, freq_ref, freq_peak=freq_peak)

    def _get_freq_scaling(self, freqs: Quantity, freq_peak: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        # Unpacking the template
        template_freq = Quantity(self.SPINNING_DUST_TEMPLATE[0], unit="GHz")
        template_amp = Quantity(self.SPINNING_DUST_TEMPLATE[1], unit="Jy/sr")
        template_amp = template_amp.to(
            "uK", equivalencies=brightness_temperature(template_freq)
        )

        peak_scale = 30 * Unit("GHz") / freq_peak

        # AME is undefined at outside of this frequency range
        if not np.logical_and(
            (freqs * peak_scale) > template_freq.min(),
            (freqs * peak_scale) < template_freq.max(),
        ).all():
            return Quantity(0)

        interp = np.interp(
            (freqs * peak_scale).si.value,
            template_freq.si.value,
            template_amp.si.value,
        )
        interp_ref = np.interp(
            (self.freq_ref * peak_scale).si.value,
            template_freq.si.value,
            template_amp.si.value,
        )
        scaling = interp / interp_ref

        return scaling


class Radio(PointSourceComponent):
    """Radio component."""

    label = "dust"

    # The radio catalog must have shape (2, `npointsources`), where
    # `npointsources` must match the number of poitns in `amp`
    catalog: np.ndarray = np.loadtxt(RADIO_CATALOG, usecols=(0, 1)).transpose()

    def __init__(self, amp: Quantity, freq_ref: Quantity, alpha: Quantity) -> None:
        super().__init__(self.label, self.catalog, amp, freq_ref, alpha=alpha)

    def _get_freq_scaling(self, freqs: Quantity, alpha: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        scaling = (freqs / self.freq_ref) ** (alpha - 2)

        return scaling
