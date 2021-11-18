import pytest

from astropy.units import Quantity, UnitsError
import numpy as np
import healpy as hp

from cosmoglobe.sky.components.synchrotron import PowerLaw
from cosmoglobe.sky.components.dust import ModifiedBlackbody
from cosmoglobe.sky._exceptions import NsideError

amp_1 = Quantity(np.ones((1, hp.nside2npix(32))), unit="K_RJ")
amp_3 = Quantity(np.ones((3, hp.nside2npix(32))), unit="K_RJ")
freq_ref_1 = Quantity([[40]], unit="GHz")
freq_ref_3 = Quantity([[40], [50], [50]], unit="GHz")
beta_1 = Quantity(np.random.randint(-2, 3, (1, hp.nside2npix(32))))
beta_3 = Quantity(np.random.randint(-2, 3, (3, hp.nside2npix(32))))
T_1 = Quantity(np.random.randint(-2, 3, (1, hp.nside2npix(32))), unit="K")
T_3 = Quantity(np.random.randint(-2, 3, (3, hp.nside2npix(32))), unit="K")


def test_radio_catalog(radio):
    """Tests the radio catalog."""

    assert radio.catalog.shape == (2, radio.amp.size)


def test_freq_ref_type():
    """Tests the type of freq_ref."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(amp_1, freq_ref_1, beta=beta_1)

    with pytest.raises(TypeError):
        PowerLaw(amp_1, np.array([[10]]), beta=beta_1)


def test_freq_ref_shape():
    """Tests that the freq_ref shape is valid."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(amp_3, freq_ref_3, beta=beta_3)

    with pytest.raises(ValueError):
        PowerLaw(amp_1, Quantity([50], unit="GHz"), beta=beta_1)

    with pytest.raises(ValueError):
        PowerLaw(amp_1, Quantity([50, 30, 10], unit="GHz"), beta=beta_1)


def test_freq_ref_unit():
    """Tests that the unit of freq_ref is compatible with GHz."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)

    with pytest.raises(UnitsError):
        PowerLaw(amp_1, Quantity([[30]], unit="s"), beta=beta_1)


def test_amp_type():
    """Tests the type of amp."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)

    with pytest.raises(TypeError):
        PowerLaw(np.ones((1, hp.nside2npix(32))), freq_ref_1, beta=beta_1)


def test_amp_shape():
    """Tests that the amp shape is valid."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(amp_3, freq_ref_3, beta=beta_3)

    with pytest.raises(NsideError):
        PowerLaw(Quantity(np.ones((1, 23423)), unit="K_RJ"), freq_ref_1, beta=beta_1)

    with pytest.raises(ValueError):
        PowerLaw(
            Quantity(np.ones((2, hp.nside2npix(32))), unit="K_RJ"), freq_ref_1, beta=beta_1
        )

    with pytest.raises(ValueError):
        PowerLaw(
            Quantity(np.ones((hp.nside2npix(32))), unit="K_RJ"), freq_ref_1, beta=beta_1
        )


def test_amp_unit():
    """Tests that the unit of amp is compatible with uK."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(
        Quantity(np.ones((1, hp.nside2npix(32))), unit="Jy/sr"), freq_ref_1, beta=beta_1
    )

    with pytest.raises(UnitsError):
        PowerLaw(
            Quantity(np.ones((1, hp.nside2npix(32))), unit="m"), freq_ref_1, beta=beta_1
        )


def test_spectral_params_type():
    """Tests the type of spectral_params."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    PowerLaw(
        amp_3,
        freq_ref_3,
        beta=beta_3,
    )
    ModifiedBlackbody(amp_3, freq_ref_3, beta=beta_3, T=T_3)

    with pytest.raises(TypeError):
        ModifiedBlackbody(amp_1, freq_ref_1, beta=beta_1, T=[[1]])

    with pytest.raises(ValueError):
        ModifiedBlackbody(
            amp_1,
            freq_ref_1,
            beta=beta_1,
            T=Quantity(np.random.randint(-2, 3, (3, 12312)), unit="K"),
        )


def test_spectral_params_shape():
    """Tests the shape of spectral_params."""

    PowerLaw(amp_1, freq_ref_1, beta=beta_1)
    ModifiedBlackbody(
        amp_3,
        freq_ref_3,
        beta=Quantity([[10], [40], [20]]),
        T=Quantity([[10], [1], [1]], unit="K"),
    )
    ModifiedBlackbody(
        amp_3,
        freq_ref_3,
        beta=Quantity([[10], [40], [20]]),
        T=T_3,
    )
    PowerLaw(
        amp_3,
        freq_ref_3,
        beta=Quantity(np.zeros((3, hp.nside2npix(32))), unit="K"),
    )
    PowerLaw(
        amp_1,
        freq_ref_1,
        beta=Quantity(np.zeros((1, hp.nside2npix(32))), unit="K"),
    )

    with pytest.raises(ValueError):
        PowerLaw(amp_1, freq_ref_1, beta=Quantity(np.zeros((hp.nside2npix(32)))))

    with pytest.raises(ValueError):
        ModifiedBlackbody(
            amp_1,
            freq_ref_1,
            beta=Quantity(np.zeros(3)),
            T=Quantity(np.zeros((3, 1))),
        )
