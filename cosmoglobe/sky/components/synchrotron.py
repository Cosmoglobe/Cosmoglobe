from astropy.units import Quantity

from cosmoglobe.sky.base_components import DiffuseComponent


class Synchrotron(DiffuseComponent):
    r"""Class representing the synchrotron component in the sky model.

    Notes
    -----
    This is a generic power law given at a reference frequency
    :math:`\nu_{s,0}` with a power law :math:`\beta` in Rayleigh-Jeans
    temperature. It is defined using the convention in
    `BeyondPlanck (2020), Section 3.3.1
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_\mathrm{RJ}^{\mathrm{s}}(\nu) \propto
        \left( \frac{\nu}{\nu_\mathrm{0,s}} \right)^
        {\beta + C \ln \nu / \nu_{0,s}},

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission. :math:`C` is set to 0 for all current implementations
    as of BP9.
    """

    label = "synch"

    def __init__(self, amp: Quantity, freq_ref: Quantity, beta: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref, beta=beta)

    def get_freq_scaling(self, freqs: Quantity, beta: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        return (freqs / self.freq_ref) ** beta
