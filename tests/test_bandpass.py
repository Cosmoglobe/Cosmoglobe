from astropy.units.core import UnitsError
import pytest

from astropy.units import Unit, UnitTypeError
import numpy as np

from cosmoglobe.sky._bandpass import get_bandpass_coefficient, get_normalized_bandpass


def test_normalized():
    """Test normalized bandpass."""

    freqs = np.arange(1, 101) * Unit("GHz")
    bandpass = np.random.randint(10, 100, (100,)) * Unit("uK")

    assert get_normalized_bandpass(freqs, bandpass).unit == Unit("1/GHz")
    assert np.trapz(get_normalized_bandpass(freqs, bandpass), freqs).value == pytest.approx(1)


def test_coeff():
    """Test that the coeff works."""

    freqs = [30, 40] * Unit("GHz")
    bandpass = [0.5, 0.5] * Unit("1/GHz")
    get_bandpass_coefficient(freqs, bandpass, Unit("uK"), Unit("MJy/sr"))
    get_bandpass_coefficient(freqs, bandpass, "uK", "MJy/sr")
    with pytest.raises(UnitTypeError):
        get_bandpass_coefficient(freqs, bandpass, Unit("GHz"), Unit("MJy/sr"))
