from astropy.units import Quantity
import numpy as np

from cosmoglobe.data import DATA_DIR
from cosmoglobe.sky.base_components import PointSourceComponent


RADIO_CATALOG = DATA_DIR / "radio_catalog.dat"


class Radio(PointSourceComponent):
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

    label = "radio"

    # The radio catalog must have shape (2, `npointsources`), where
    # `npointsources` must match the number of poitns in `amp`
    catalog: np.ndarray = np.loadtxt(RADIO_CATALOG, usecols=(0, 1)).transpose()

    def __init__(self, amp: Quantity, freq_ref: Quantity, alpha: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, self.catalog, amp, freq_ref, alpha=alpha)

    def get_freq_scaling(self, freqs: Quantity, alpha: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        scaling = (freqs / self.freq_ref) ** (alpha - 2)

        return scaling
