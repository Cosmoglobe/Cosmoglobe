from typing import Tuple

import numpy as np
from astropy.units import Quantity, Unit

from cosmoglobe.data import DATA_DIR
from cosmoglobe.sky._base_components import DiffuseComponent
from cosmoglobe.sky._units import cmb_equivalencies
from cosmoglobe.sky.components._labels import SkyComponentLabel

SPDUST2_FILE = DATA_DIR / "spdust2_cnm.dat"

SPDUST2_FREQS, SPDUST2_AMPS = np.loadtxt(SPDUST2_FILE, unpack=True)
SPDUST2_FREQS *= Unit("GHz")
SPDUST2_AMPS *= Unit("Jy/sr")
SPDUST2_TEMPLATE = (SPDUST2_FREQS, SPDUST2_AMPS)


class SpinningDust(DiffuseComponent):
    r"""Class representing the AME component in the Cosmoglobe Sky Model.

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

    label = SkyComponentLabel.AME
    freq_range = (0 * Unit("Hz"), 500 * Unit("GHz"))
    SPINNING_DUST_TEMPLATE: Tuple[Quantity, Quantity] = SPDUST2_TEMPLATE

    def get_freq_scaling(self, freqs: Quantity, freq_peak: Quantity) -> Quantity:
        """See base class."""

        spdust2_freqs = SPDUST2_TEMPLATE[0]
        # We set the spdust2 template amplitudes to the component amplitude units.
        spdust2_amps = SPDUST2_TEMPLATE[1].to(
            self.amp.unit, equivalencies=cmb_equivalencies(spdust2_freqs)
        )

        peak_scale = 30 * Unit("GHz") / freq_peak
        interp = np.interp(
            (freqs * peak_scale).si.value,
            spdust2_freqs.si.value,
            spdust2_amps.decompose().value,
            left=0.0,
            right=0.0,
        )
        interp_ref = np.interp(
            (self.freq_ref * peak_scale).si.value,
            spdust2_freqs.si.value,
            spdust2_amps.decompose().value,
        )

        scaling = interp / interp_ref

        return Quantity(scaling)
