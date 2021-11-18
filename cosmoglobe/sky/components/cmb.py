from astropy.units import Quantity, Unit

from cosmoglobe.sky._base_components import DiffuseComponent
from cosmoglobe.sky.components._labels import SkyComponentLabel


class CMB(DiffuseComponent):
    r"""Class representing the CMB component in the Cosmoglobe Sky Model.

    Notes
    -----
    The CMB emission is defined using the convention in
    `BeyondPlanck (2020), Section 3.2
    <https://arxiv.org/pdf/2011.05609.pdf>`_;

    .. math::

        \boldsymbol{s}_{\mathrm{RJ}}^{\mathrm{CMB}}(\nu) \propto
        \frac{x^{2} \mathrm{e}^{x}}{\left(\mathrm{e}^{x}-1\right)
        ^{2}} \boldsymbol{s}^{\mathrm{CMB}},

    where :math:`\nu` is the frequency for which we are simulating the
    sky emission, :math:`x=h \nu / k T_{0}` and
    :math:`T_0 = 2.7255 \mathrm{K}` as of BP9.
    """

    label = SkyComponentLabel.CMB
    freq_range = (0 * Unit("Hz"), 1 * Unit("THz"))

    def get_freq_scaling(self, *_) -> Quantity:
        """See base class.

        NOTE: The CMB amplitude maps are stored in units of uK_CMB for which
        they are constant over frequencies.
        """

        return Quantity(1)
