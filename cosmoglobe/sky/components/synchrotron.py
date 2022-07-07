from astropy.units import Quantity, Unit

from cosmoglobe.sky._base_components import DiffuseComponent
from cosmoglobe.sky.components._labels import SkyComponentLabel


class PowerLaw(DiffuseComponent):
    r"""Class representing the Synchrotron component in the sky model.

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

    label = SkyComponentLabel.SYNCH
    freq_range = (0 * Unit("Hz"), 2 * Unit("THz"))

    def get_freq_scaling(self, freqs: Quantity, beta: Quantity) -> Quantity:
        """See base class."""

        return (freqs / self.freq_ref) ** beta
