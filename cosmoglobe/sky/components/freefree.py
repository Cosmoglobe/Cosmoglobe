from astropy.units import Quantity, Unit, quantity_input
import numpy as np

from cosmoglobe.sky._base_components import DiffuseComponent
from cosmoglobe.sky.components._labels import SkyComponentLabel


class LinearOpticallyThin(DiffuseComponent):
    r"""Class representing the free-free component in the Cosmoglobe Sky Model.

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

    label = SkyComponentLabel.FF
    freq_range = (0 * Unit("Hz"), 5 * Unit("THz"))

    def get_freq_scaling(self, freqs: Quantity, T_e: Quantity) -> Quantity:
        """See base class."""

        gaunt_factor_ratio = gaunt_factor(freqs, T_e) / gaunt_factor(self.freq_ref, T_e)
        scaling = (self.freq_ref / freqs) ** 2 * gaunt_factor_ratio

        return scaling


@quantity_input(freq="Hz", T_e="K")
def gaunt_factor(freq: Quantity, T_e: Quantity) -> Quantity:
    """Returns the Gaunt factor.

    Computes the gaunt factor for a given frequency and electron
    temperaturein SI units.

    Parameters
    ----------
    freq
        Frequency [Hz].
    T_e
        Electron temperature [K].

    Returns
    -------
    gaunt_factor
        Gaunt Factor.
    """

    # Avoiding overflow and underflow.
    T_e = T_e.astype(np.float64)
    T_e = (T_e.to("kK")).value / 10
    freq = (freq.to("GHz")).value

    gaunt_factor = np.log(
        np.exp(5.96 - (np.sqrt(3) / np.pi) * np.log(freq * T_e ** -1.5)) + np.e
    )

    return Quantity(gaunt_factor)
