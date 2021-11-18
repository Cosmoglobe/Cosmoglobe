import pytest

from astropy.units import Quantity, Unit, UnitsError
import numpy as np
import healpy as hp

from cosmoglobe.sky.components.ame import SpinningDust
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.dust import ModifiedBlackbody
from cosmoglobe.sky.components.freefree import LinearOpticallyThin
from cosmoglobe.sky.components.synchrotron import PowerLaw
from cosmoglobe.sky.components.radio import AGNPowerLaw


def test_init_synch():
    """Test the initialization of Synchrotron."""
    amp_1 = Quantity(np.ones((1, hp.nside2npix(32))), unit="K_RJ")
    amp_3 = Quantity(np.ones((3, hp.nside2npix(32))), unit="K_RJ")
    freq_ref_1 = Quantity([[40]], unit="GHz")
    freq_ref_3 = Quantity([[40], [40], [40]], unit="GHz")
    beta_1 = Quantity(np.ones((1, hp.nside2npix(32))))
    beta_3 = Quantity(np.ones((3, hp.nside2npix(32))))

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(amp_3, freq_ref_3, beta=beta_3)


def test_init_dust():
    """Test the initialization of Dust."""
    amp_1 = Quantity(np.ones((1, hp.nside2npix(32))), unit="K_RJ")
    amp_3 = Quantity(np.ones((3, hp.nside2npix(32))), unit="K_RJ")
    freq_ref_1 = Quantity([[40]], unit="GHz")
    freq_ref_3 = Quantity([[40], [40], [40]], unit="GHz")
    beta_1 = Quantity(np.ones((1, hp.nside2npix(32))))
    beta_3 = Quantity(np.ones((3, hp.nside2npix(32))))
    T_1 = Quantity([[1]])
    T_3 = Quantity([[3], [2], [2]])

    ModifiedBlackbody(amp_1, freq_ref_1, beta=beta_1, T=T_1)
    ModifiedBlackbody(amp_3, freq_ref_3, beta=beta_3, T=T_3)


def test_init_ff():
    """Test the initialization of ff."""
    amp_1 = Quantity(np.ones((1, hp.nside2npix(32))), unit="K_RJ")
    amp_3 = Quantity(np.ones((3, hp.nside2npix(32))), unit="K_RJ")
    freq_ref_1 = Quantity([[40]], unit="GHz")
    freq_ref_3 = Quantity([[40], [40], [40]], unit="GHz")
    T_e_1 = Quantity(np.ones((1, hp.nside2npix(32))))
    T_e_3 = Quantity(np.ones((3, hp.nside2npix(32))))

    LinearOpticallyThin(amp_1, freq_ref_1, T_e=T_e_1)
    LinearOpticallyThin(amp_3, freq_ref_3, T_e=T_e_3)


def test_init_radio():
    """Test the initialization of Radio."""
    amp = Quantity(np.ones((1, 12192)), unit="mJy")
    freq_ref = Quantity([[40]], unit="GHz")
    alpha = Quantity(np.ones((1, 12192)))

    AGNPowerLaw(amp, freq_ref, alpha=alpha)

    with pytest.raises(TypeError):
        AGNPowerLaw(amp.value, freq_ref, alpha=alpha)

    with pytest.raises(ValueError):
        AGNPowerLaw(Quantity(np.ones((3, 12192)), unit="mJy"), freq_ref, alpha=alpha)

    with pytest.raises(ValueError):
        AGNPowerLaw(Quantity(np.ones((1, 1210)), unit="mJy"), freq_ref, alpha=alpha)

    with pytest.raises(UnitsError):
        AGNPowerLaw(Quantity(np.ones((1, 12192)), unit="mJy/sr"), freq_ref, alpha=alpha)

    with pytest.raises(TypeError):
        AGNPowerLaw(amp, freq_ref, alpha=np.ones((1, 12192)))

    with pytest.raises(ValueError):
        AGNPowerLaw(amp, freq_ref, alpha=Quantity(np.ones((3, 12192))))


def test_radio_unit(radio):
    """Tests the unit of radio."""

    assert radio.amp.unit == Unit("mJy")


def test_init_ame():
    """Test the initialization of Radio."""
    amp = Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="uK_RJ")
    freq_ref = Quantity([[40], [40], [40]], unit="GHz")
    freq_peak = Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="GHz")

    SpinningDust(amp, freq_ref, freq_peak=freq_peak)


def test_init_cmb():
    """Test the initialization of Radio."""
    amp = Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="uK_CMB")
    freq_ref = Quantity([[40], [40], [40]], unit="GHz")

    CMB(amp, freq_ref)
