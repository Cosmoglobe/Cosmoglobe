from cosmoglobe.sky.components.dust import ThermalDust
import pytest

from astropy.units import Unit, Quantity, UnitsError
import numpy as np
import healpy as hp

from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky.model import SkyModel


def test_skymodel(sky_model):
    """Tests the sky sky_model."""
    sky_model(100 * Unit("GHz"))
    sky_model([100, 102, 104] * Unit("GHz"))
    sky_model([100, 102, 104] * Unit("GHz"))
    sky_model([100, 102, 104] * Unit("GHz"), [3, 10, 3] * Unit("uK"))
    sky_model(
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
    )
    sky_model(
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
        output_unit="MJy/sr",
    )


def test_pointsource(sky_model):
    """Tests the sky sky_model."""

    emission = sky_model(100 * Unit("GHz"), components=["radio"])
    assert emission.shape == (1, hp.nside2npix(sky_model.nside))

    sky_model([100, 102, 104] * Unit("GHz"), components=["radio"])
    sky_model(
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        components=["radio"],
    )
    sky_model(
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
        components=["radio"],
    )
    assert (
        sky_model(
            [100, 102, 104] * Unit("GHz"),
            [3, 10, 3] * Unit("uK"),
            fwhm=30 * Unit("arcmin"),
            output_unit="MJy/sr",
            components=["radio"],
        ).shape
        == (1, hp.nside2npix(sky_model.nside))
    )


@pytest.mark.parametrize("nside", [16, 32, 64, 128, 256, 512, 1024])
def test_nside(nside):
    """Tests a set of nsides."""

    dust = ThermalDust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(nside))), unit="K"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(nside)))),
        T=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(nside))), unit="K"),
    )
    sky_model = SkyModel(nside, {"dust":dust})
    sky_model(100 * Unit("GHz"))
    sky_model([100, 120] * Unit("GHz"), bandpass=[1, 3] * Unit("MJy/sr"))


def test_bp_integ(dust0spec, dust1spec, dust2spec):
    """Tests bp integration for 0, 1, 2 spatially varying spectral parameters."""

    comps = {"d1": dust0spec, "d2": dust1spec, "d3": dust2spec}
    sky_model = SkyModel(32, comps)
    emission1 = sky_model([100, 120] * Unit("GHz"), bandpass=[1, 3] * Unit("uK"))
    emission2 = sky_model(
        freqs=[100, 120] * Unit("GHz"),
        bandpass=[1, 3] * Unit("uK"),
        output_unit="MJy/sr",
    )
    for comp in comps.values():
        assert emission1.shape == comp.amp.shape
        assert emission2.unit == Unit("MJy/sr")


def test_inputs(sky_model):
    """Test the input parameters to sky_model"""

    with pytest.raises(UnitsError):
        sky_model(10 * Unit("m"))

    with pytest.raises(TypeError):
        sky_model(10)

    with pytest.raises(UnitsError):
        sky_model([10, 11, 12] * Unit("m"))

    with pytest.raises(TypeError):
        sky_model([10, 11, 12])

    with pytest.raises(UnitsError):
        sky_model(
            [10, 11, 12] * Unit("GHz"),
            bandpass=[10, 11, 12] * Unit("GHz"),
        )

    with pytest.raises(ValueError):
        sky_model(
            [10, 11] * Unit("GHz"),
            bandpass=[11, 12, 13] * DEFAULT_OUTPUT_UNIT,
        )
