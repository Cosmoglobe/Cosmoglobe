from pathlib import Path

from astropy.units import Quantity, Unit, brightness_temperature
import numpy as np

from cosmoglobe.sky.base_components import DiffuseComponent, PointSourceComponent
from cosmoglobe.utils.functions import (
    blackbody_emission,
    gaunt_factor,
    thermodynamical_to_brightness,
)

DATA_DIR = Path(__file__).parent.parent.resolve() / "data"
RADIO_CATALOG = DATA_DIR / "radio_catalog.dat"
SPDUST2_FILE = DATA_DIR / "spdust2_cnm.dat"


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


class AME(DiffuseComponent):
    r"""Class representing the AME component in the sky model.

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

    label = "ame"

    SPINNING_DUST_TEMPLATE = np.loadtxt(SPDUST2_FILE).transpose()

    def __init__(self, amp: Quantity, freq_ref: Quantity, freq_peak: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref, freq_peak=freq_peak)

    def get_freq_scaling(self, freqs: Quantity, freq_peak: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        # Unpacking the template
        template_freq = Quantity(self.SPINNING_DUST_TEMPLATE[0], unit="GHz")
        template_amp = Quantity(self.SPINNING_DUST_TEMPLATE[1], unit="Jy/sr")
        template_amp = template_amp.to(
            "uK", equivalencies=brightness_temperature(template_freq)
        )

        peak_scale = 30 * Unit("GHz") / freq_peak

        # AME is undefined at outside of this frequency range
        if not np.logical_and(
            freqs > template_freq.min(),
            freqs < template_freq.max(),
        ).all():
            return Quantity(0)

        interp = np.interp(
            (freqs * peak_scale).si.value,
            template_freq.si.value,
            template_amp.si.value,
        )
        interp_ref = np.interp(
            (self.freq_ref * peak_scale).si.value,
            template_freq.si.value,
            template_amp.si.value,
        )
        scaling = interp / interp_ref

        return scaling


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


class CMB(DiffuseComponent):
    r"""Class representing the CMB component.

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

    label = "cmb"

    def __init__(self, amp: Quantity, freq_ref: Quantity) -> None:
        """Initializing base class."""

        super().__init__(self.label, amp, freq_ref)

    def get_freq_scaling(self, freqs: Quantity) -> Quantity:  # type: ignore
        """See base class."""

        return np.expand_dims(thermodynamical_to_brightness(freqs), axis=0)


COSMOGLOBE_COMPS = {
    comp.label: comp  # type: ignore
    for comp in [
        AME,
        CMB,
        Dust,
        FreeFree,
        Radio,
        Synchrotron,
    ]
}
