from astropy.units import Quantity, Unit, brightness_temperature
import numpy as np

from cosmoglobe.data import DATA_DIR
from cosmoglobe.sky.base_components import DiffuseComponent


SPDUST2_FILE = DATA_DIR / "spdust2_cnm.dat"


class AME(DiffuseComponent):
    r"""Class representing the AME component in the sky model.

    Notes
    -----
    The spinning dust emission is defined using the convention in
    `BeyondPlanck (2020), Section 3.3.4
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{sd}}(\nu) \propto
        \nu^{-2} \boldsymbol{s}_{0}^{\mathrm{sd}}\left(\nu \cdot
        \frac{30.0\; \mathrm{GHz}}{\nu_{p}}\right)

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission, :math:`\boldsymbol{s}_0^{\mathrm{sd}}` is the `spdust2`
    template, and :math:`\nu_p` the peak frequency.
    """

    label = "ame"

    SPINNING_DUST_TEMPLATE = np.loadtxt(SPDUST2_FILE).transpose()

    def __init__(self, amp: Quantity, freq_ref: Quantity, freq_peak: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref, freq_peak=freq_peak)

    def get_freq_scaling(self, freqs: Quantity, freq_peak: Quantity) -> Quantity:  # type: ignore
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
            freqs > template_freq.min(),
            freqs < template_freq.max(),
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
