import pytest

from astropy.units import Unit
import numpy as np

from cosmoglobe.sky._units import *
from cosmoglobe.sky._bandpass import get_bandpass_coefficient, get_normalized_weights


def test_normalized():
    """Test normalized bandpass."""

    freqs = np.arange(1, 101) * Unit("GHz")
    weights_RJ = np.arange(1, 101) * 2 * Unit("uK_RJ")
    weights_CMB = np.arange(1, 101) * 2 * Unit("K_CMB")
    weights_JY = np.arange(1, 101) * 2 * Unit("Jy/sr")

    assert get_normalized_weights(freqs, weights_RJ, Unit("uK_CMB")).unit.is_equivalent("1/GHz")
    assert get_normalized_weights(freqs, weights_CMB, Unit("uK_CMB")).unit.is_equivalent("1/GHz")
    assert get_normalized_weights(freqs, weights_JY, Unit("uK_CMB")).unit.is_equivalent("1/GHz")

    assert np.trapz(
        get_normalized_weights(freqs, weights_RJ, Unit("uK_CMB")), freqs
        ).value == pytest.approx(1)


def test_coeff():
    """Test that the coeff works."""

    freqs = [30, 40] * Unit("GHz")
    weights = [0.5, 0.5]  * Unit("uK_RJ")
    get_bandpass_coefficient(freqs, weights, Unit("uK_RJ"), Unit("MJy/sr"))
    with pytest.raises(KeyError):
        get_bandpass_coefficient(freqs, weights, Unit("GHz"), Unit("MJy/sr"))

    freqs = np.arange(1, 101) * Unit("GHz")
    weights = np.arange(1, 101) * 2 * Unit("uK_RJ")
    weights_RJ = get_normalized_weights(freqs, weights, Unit("uK_RJ"))
    coeff = get_bandpass_coefficient(freqs, weights_RJ, Unit("uK_RJ"), Unit("uK_CMB"))
    assert coeff.unit.is_equivalent("uK_CMB/uK_RJ")
