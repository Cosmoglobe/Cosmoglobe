from cosmoglobe.sky.components.dust import ThermalDust
import pytest

from astropy.units import Unit, Quantity, UnitsError
import numpy as np
import healpy as hp

from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky.simulator import DEFAULT_SIMULATOR as simulator


def test_1_comp(synch_1):
    """Tests the sky simulator."""
    simulator(32, [synch_1], 100 * Unit("GHz"))
    simulator(32, [synch_1], [100, 102, 104] * Unit("GHz"))
    simulator(32, [synch_1], [100, 102, 104] * Unit("GHz"))
    simulator(32, [synch_1], [100, 102, 104] * Unit("GHz"), [3, 10, 3] * Unit("uK"))
    simulator(
        32,
        [synch_1],
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
    )
    simulator(
        32,
        [synch_1],
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
        output_unit="MJy/sr",
    )


def test_multi_comsp(synch_1, dust_3):
    """Tests the sky simulator."""

    simulator(32, [synch_1, dust_3], 100 * Unit("GHz"))
    simulator(32, [synch_1, dust_3], [100, 102, 104] * Unit("GHz"))
    simulator(32, [synch_1, dust_3], [100, 102, 104] * Unit("GHz"))
    simulator(
        32, [synch_1, dust_3], [100, 102, 104] * Unit("GHz"), [3, 10, 3] * Unit("uK")
    )
    simulator(
        32,
        [synch_1, dust_3],
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
    )
    simulator(
        32,
        [synch_1, dust_3],
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
        output_unit="MJy/sr",
    )


def test_pointsource(radio):
    """Tests the sky simulator."""

    emission = simulator(128, [radio], 100 * Unit("GHz"))
    assert emission.shape == (1, hp.nside2npix(128))

    simulator(128, [radio], [100, 102, 104] * Unit("GHz"))
    simulator(128, [radio], [100, 102, 104] * Unit("GHz"), [3, 10, 3] * Unit("uK"))
    simulator(
        128,
        [radio],
        [100, 102, 104] * Unit("GHz"),
        [3, 10, 3] * Unit("uK"),
        fwhm=30 * Unit("arcmin"),
    )
    assert (
        simulator(
            128,
            [radio],
            [100, 102, 104] * Unit("GHz"),
            [3, 10, 3] * Unit("uK"),
            fwhm=30 * Unit("arcmin"),
            output_unit="MJy/sr",
        ).shape
        == (1, hp.nside2npix(128))
    )


@pytest.mark.parametrize("nside", [16, 32, 64, 128, 256, 512, 1024, 2048])
def test_nside(nside):
    """Tests a set of nsides."""
    comp = ThermalDust(
        Quantity(np.ones((1, hp.nside2npix(nside))), unit="K"),
        Quantity([[40]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 100, (1, hp.nside2npix(nside)))),
        T=Quantity([[1000]], unit="K"),
    )
    simulator(nside, [comp], 100 * Unit("GHz"))
    simulator(nside, [comp], [100, 120] * Unit("GHz"), bandpass=[1, 3] * Unit("MJy/sr"))


def test_bp_integ(dust0spec, dust1spec, dust2spec):
    """Tests bp integration for 0, 1, 2 spatially varying spectral parameters."""

    comps = [dust0spec, dust1spec, dust2spec]
    for comp in comps:
        emission1 = simulator(
            32, [comp], [100, 120] * Unit("GHz"), bandpass=[1, 3] * Unit("uK")
        )
        emission2 = simulator(
            32,
            [comp],
            [100, 120] * Unit("GHz"),
            bandpass=[1, 3] * Unit("uK"),
            output_unit="MJy/sr",
        )
        assert emission1.shape == comp.amp.shape
        assert emission2.unit == Unit("MJy/sr")


def test_inputs(synch_1, dust_3):
    """Test the input parameters to simulator"""

    for comp in [synch_1, dust_3]:

        with pytest.raises(UnitsError):
            simulator(32, [comp], 10 * Unit("m"))

        with pytest.raises(TypeError):
            simulator(32, [comp], 10)

        with pytest.raises(UnitsError):
            simulator(32, [comp], [10, 11, 12] * Unit("m"))

        with pytest.raises(TypeError):
            simulator(32, [comp], [10, 11, 12])

        with pytest.raises(UnitsError):
            simulator(
                32,
                [comp],
                [10, 11, 12] * Unit("GHz"),
                bandpass=[10, 11, 12] * Unit("GHz"),
            )

        with pytest.raises(ValueError):
            simulator(
                32,
                [comp],
                [10, 11] * Unit("GHz"),
                bandpass=[11, 12, 13] * DEFAULT_OUTPUT_UNIT,
            )
