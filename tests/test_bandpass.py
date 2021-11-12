import pytest

from astropy.units import Unit
import numpy as np

from cosmoglobe.sky._units import *
from cosmoglobe.sky._bandpass import get_bandpass_coefficient, get_normalized_weights


def test_normalized():
    """Test normalized bandpass."""

    freqs = np.arange(1, 101) * Unit("GHz")
    weights = np.arange(1, 101) * 2 * Unit("uK_RJ")

    assert get_normalized_weights(freqs, weights, Unit("uK_CMB")).unit.is_equivalent("1/GHz")
    assert np.trapz(
        get_normalized_weights(freqs, weights, Unit("uK_CMB")), freqs
        ).value == pytest.approx(1)


def test_coeff():
    """Test that the coeff works."""

    freqs = [30, 40] * Unit("GHz")
    weights = [0.5, 0.5]  * Unit("uK_RJ")
    bandpass = get_normalized_weights(freqs, weights, Unit("uK_CMB"))
    get_bandpass_coefficient(freqs, weights, Unit("uK_RJ"), Unit("MJy/sr"))
    with pytest.raises(KeyError):
        get_bandpass_coefficient(freqs, weights, Unit("GHz"), Unit("MJy/sr"))
