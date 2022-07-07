from astropy.units import Quantity, Unit, quantity_input
import numpy as np

from cosmoglobe.sky._base_components import DiffuseComponent
from cosmoglobe.sky.components._labels import SkyComponentLabel
import cosmoglobe.sky._constants as const


class ModifiedBlackbody(DiffuseComponent):
    r"""Class representing the thermal dust component in the Cosmoglobe Sky Model.

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

    label = SkyComponentLabel.DUST
    freq_range = (0 * Unit("Hz"), 100 * Unit("THz"))

    def get_freq_scaling(
        self,
        freqs: Quantity,
        beta: Quantity,
        T: Quantity,
    ) -> Quantity:
        """See base class."""

        blackbody_ratio = blackbody_emission(freqs, T) / blackbody_emission(
            self.freq_ref, T
        )
        scaling = (freqs / self.freq_ref) ** (beta - 2) * blackbody_ratio

        return scaling


@quantity_input(freq="Hz", T="K")
def blackbody_emission(freq: Quantity, T: Quantity) -> Quantity:
    """Returns the blackbody emission.

    Computes the emission emitted by a blackbody with with temperature
    T at a frequency freq in SI units [W / m^2 Hz sr].

    Parameters
    ----------
    freq
        Frequency [Hz].
    T
        Temperature of the blackbody [K].

    Returns
    -------
    emission
        Blackbody emission [W / m^2 Hz sr].
    """

    # Avoiding overflow and underflow.
    T = T.astype(np.float64)

    term1 = (2 * const.h * freq ** 3) / const.c ** 2
    term2 = np.expm1((const.h * freq) / (const.k_B * T))

    emission = term1 / term2 / Unit("sr")

    return emission.to("W / m^2 Hz sr")
