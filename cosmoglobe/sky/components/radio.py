import numpy as np
from astropy.units import Quantity

from cosmoglobe.data import DATA_DIR
from cosmoglobe.sky._base_components import PointSourceComponent
from cosmoglobe.sky.components._labels import SkyComponentLabel

RADIO_CATALOG = DATA_DIR / "radio_catalog.dat"


class AGNPowerLaw(PointSourceComponent):
    r"""Class representing the radio component.

    Notes
    -----
    This is a generic power law given at a reference frequency
    :math:`\nu_{\mathrm{0, src}}` with a power law spectral index
    :math:`\alpha`. It is defined using the convention in
    `BeyondPlanck (2020), Section 3.4.1
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{src}}(\nu) \propto
        \left(\frac{\nu}{\nu_{\mathrm{0, src}}}\right)^{\alpha-2}

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission.
    """

    label = SkyComponentLabel.RADIO
    catalog = Quantity(
        np.loadtxt(RADIO_CATALOG, usecols=(0, 1)).transpose(), unit="deg"
    )

    def get_freq_scaling(self, freqs: Quantity, alpha: Quantity) -> Quantity:
        """See base class."""

        scaling = (freqs / self.freq_ref) ** (alpha - 2)

        return scaling
