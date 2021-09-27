from astropy.units import Quantity

from cosmoglobe.sky.base_components import DiffuseComponent
from cosmoglobe.utils.functions import blackbody_emission


class Dust(DiffuseComponent):
    r"""Class representing the thermal dust component in the sky model.

    Notes
    -----
    This is a modified blackbody with a power law spectral index
    :math:`\beta_\mathrm{d}` in Rayleigh-Jeans temperature, and thermal dust
    temperature :math:`T_{\mathrm{d}}`. It is defined using the convention
    in `BeyondPlanck (2020), Section 3.3.3
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{d}}(\nu) \propto
        \frac{\nu^{\beta_{\mathrm{d}}+1}}{\mathrm{e}^
        {h\nu/kT_{\mathrm{d}}}-1},

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission, :math:`h` is Planck's constant, and :math:`k` is the
    Boltzmann constant.
    """

    label = "dust"

    def __init__(
        self, amp: Quantity, freq_ref: Quantity, beta: Quantity, T: Quantity
    ) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref, beta=beta, T=T)

    def get_freq_scaling(  # type: ignore
        self, freqs: Quantity, beta: Quantity, T: Quantity
    ) -> Quantity:
        """See base class."""

        blackbody_ratio = blackbody_emission(freqs, T) / blackbody_emission(
            self.freq_ref, T
        )
        scaling = (freqs / self.freq_ref) ** (beta - 2) * blackbody_ratio

        return scaling
