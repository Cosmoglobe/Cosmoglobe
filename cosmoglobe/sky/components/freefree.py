from astropy.units import Quantity

from cosmoglobe.sky.base_components import DiffuseComponent
from cosmoglobe.utils.functions import gaunt_factor


class FreeFree(DiffuseComponent):
    r"""Class representing the free-free component in the sky model.

    Notes
    -----
    The free-free emission is defined using the convention in
    `BeyondPlanck (2020), Section 3.3.2
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{ff}}(\nu) \propto
        \frac{g_{\mathrm{ff}}(T_\mathrm{e})}{\nu^2},

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission, :math:`g_\mathrm{ff}` is the Gaunt factor, and
    :math:`T_\mathrm{e}` is the electron temperature.
    """

    label = "ff"

    def __init__(self, amp: Quantity, freq_ref: Quantity, T_e: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref, T_e=T_e)

    def get_freq_scaling(self, freqs: Quantity, T_e: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        gaunt_factor_ratio = gaunt_factor(freqs, T_e) / gaunt_factor(self.freq_ref, T_e)
        scaling = (self.freq_ref / freqs) ** 2 * gaunt_factor_ratio

        return scaling
