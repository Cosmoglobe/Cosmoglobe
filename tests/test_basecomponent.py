import pytest

from astropy.units import Quantity, UnitsError
import numpy as np
import healpy as hp

from cosmoglobe.sky.components import Synchrotron, Dust
from cosmoglobe.sky._exceptions import NsideError

amp_1 = Quantity(np.ones((1, hp.nside2npix(32))), unit="K")
amp_3 = Quantity(np.ones((3, hp.nside2npix(32))), unit="K")
freq_ref_1 = Quantity([[40]], unit="GHz")
freq_ref_3 = Quantity([[40], [50], [50]], unit="GHz")
beta_1 = Quantity(np.random.randint(-2, 3, (1, hp.nside2npix(32))))
beta_3 = Quantity(np.random.randint(-2, 3, (3, hp.nside2npix(32))))
T_1 = Quantity(np.random.randint(-2, 3, (1, hp.nside2npix(32))), unit="K")
T_3 = Quantity(np.random.randint(-2, 3, (3, hp.nside2npix(32))), unit="K")


def test_freq_ref_type():
    """Tests the type of freq_ref."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(amp_1, freq_ref_1, beta=beta_1)

    with pytest.raises(TypeError):
        Synchrotron(amp_1, np.array([[10]]), beta=beta_1)


def test_freq_ref_shape():
    """Tests that the freq_ref shape is valid."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(amp_3, freq_ref_3, beta=beta_3)

    with pytest.raises(ValueError):
        Synchrotron(amp_1, Quantity([50], unit="GHz"), beta=beta_1)

    with pytest.raises(ValueError):
        Synchrotron(amp_1, Quantity([50, 30, 10], unit="GHz"), beta=beta_1)


def test_freq_ref_unit():
    """Tests that the unit of freq_ref is compatible with GHz."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(amp_1, Quantity([[30]], unit="nm"), beta=beta_1)

    with pytest.raises(UnitsError):
        Synchrotron(amp_1, Quantity([[30]], unit="s"), beta=beta_1)


def test_amp_type():
    """Tests the type of amp."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)

    with pytest.raises(TypeError):
        Synchrotron(np.ones((1, hp.nside2npix(32))), freq_ref_1, beta=beta_1)


def test_amp_shape():
    """Tests that the amp shape is valid."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(amp_3, freq_ref_3, beta=beta_3)

    with pytest.raises(NsideError):
        Synchrotron(Quantity(np.ones((1, 23423)), unit="K"), freq_ref_1, beta=beta_1)

    with pytest.raises(ValueError):
        Synchrotron(
            Quantity(np.ones((2, hp.nside2npix(32))), unit="K"), freq_ref_1, beta=beta_1
        )

    with pytest.raises(ValueError):
        Synchrotron(
            Quantity(np.ones((hp.nside2npix(32))), unit="K"), freq_ref_1, beta=beta_1
        )


def test_amp_unit():
    """Tests that the unit of amp is compatible with uK."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(Quantity(np.ones((1, hp.nside2npix(32))), unit="Jy/sr"), freq_ref_1, beta=beta_1)

    with pytest.raises(UnitsError):
        Synchrotron(Quantity(np.ones((1, hp.nside2npix(32))), unit="m"), freq_ref_1, beta=beta_1)


def test_spectral_params_type():
    """Tests the type of spectral_params."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Synchrotron(
        amp_3,
        freq_ref_3,
        beta=beta_3,
    )
    Dust(
        amp_3,
        freq_ref_3,
        beta=beta_3,
        T=T_3
    )

    with pytest.raises(TypeError):
        Dust(amp_1, freq_ref_1, beta=beta_1, T=[[1]])


def test_spectral_params_shape():
    """Tests the shape of spectral_params."""

    Synchrotron(amp_1, freq_ref_1, beta=beta_1)
    Dust(
        amp_3,
        freq_ref_3,
        beta=Quantity([[10], [40], [20]]),
        T=Quantity([[10], [1], [1]], unit="K"),
    )
    Dust(
        amp_3,
        freq_ref_3,
        beta=Quantity([[10], [40], [20]]),
        T=T_3,
    )
    Synchrotron(
        amp_3,
        freq_ref_3,
        beta=Quantity(np.zeros((3, hp.nside2npix(32))), unit="K"),
    )
    Synchrotron(
        amp_1,
        freq_ref_1,
        beta=Quantity(np.zeros((1, hp.nside2npix(32))), unit="K"),
    )

    with pytest.raises(ValueError):
        Synchrotron(amp_1, freq_ref_1, beta=Quantity(np.zeros((hp.nside2npix(32)))))

    with pytest.raises(ValueError):
        Dust(
            amp_1,
            freq_ref_1,
            beta=Quantity(np.zeros(3)),
            T=Quantity(np.zeros((3, 1))),
        )
