import pytest

from astropy.units import Unit

from cosmoglobe.sky import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky.components import Synchrotron
from cosmoglobe.sky.simulation_strategy import DiffuseSimulationStrategy, get_simulation_strategy

fwhm = 30 * Unit("arcmin")

def test_wrong_comp():
    """tests that we raise error when a non sky comp is used."""
    with pytest.raises(NotImplementedError):
        get_simulation_strategy(1)

def test_diffuse_delta(synch_1, synch_3):
    """Test delta simulation for diffuse comps."""
    synchs = [synch_1, synch_3]

    for synch in synchs:
        protocol = DiffuseSimulationStrategy()
        protocol.delta(synch, 10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT)
        protocol.delta(synch, 10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT)
        protocol.delta(synch, 10 * Unit("GHz"), output_unit=Unit("MJy/sr"))
        protocol.delta(synch, 10 * Unit("GHz"), output_unit=Unit("MJy/sr"))

        assert (
            protocol.delta(synch, 10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT).shape
            == synch.amp.shape
        )
        assert (
            protocol.delta(synch, 10 * Unit("GHz"), output_unit="MJy/sr").shape
            == synch.amp.shape
        )


def test_diffuse_bandpass(synch_1, synch_3):
    """Test delta simulation for diffuse comps."""

    synchs = [synch_1, synch_3]

    for synch in synchs:
        protocol = DiffuseSimulationStrategy()
        protocol.bandpass(synch, [10, 11, 12] * Unit("GHz"), None, DEFAULT_OUTPUT_UNIT)
        protocol.bandpass(synch, [10, 11, 12] * Unit("GHz"), None, DEFAULT_OUTPUT_UNIT)
        protocol.bandpass(
            synch, [10, 11, 12] * Unit("GHz"), None, output_unit=Unit("MJy/sr")
        )
        protocol.bandpass(
            synch,
            [10, 11, 12] * Unit("GHz"),
            bandpass=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=DEFAULT_OUTPUT_UNIT,
        )
        protocol.bandpass(
            synch,
            [10, 11, 12] * Unit("GHz"),
            bandpass=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )
        protocol.bandpass(
            synch,
            [10, 11, 12] * Unit("GHz"),
            bandpass=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )
        protocol.bandpass(
            synch,
            [10, 11, 12] * Unit("GHz"),
            bandpass=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )

        assert (
            protocol.bandpass(synch, [10, 11] * Unit("GHz"), None, DEFAULT_OUTPUT_UNIT).shape
            == synch.amp.shape
        )
        assert (
            protocol.bandpass(
                synch,
                [10, 11] * Unit("GHz"),
                bandpass=[11, 13] * DEFAULT_OUTPUT_UNIT,
                output_unit="MJy/sr",
            ).shape
            == synch.amp.shape
        )
