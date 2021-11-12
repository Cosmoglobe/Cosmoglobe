import pytest

from astropy.units import Unit

from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT

fwhm = 30 * Unit("arcmin")


def test_diffuse_delta(synch_1, synch_3):
    """Test delta simulation for diffuse comps."""
    synchs = [synch_1, synch_3]

    for synch in synchs:
        synch.get_delta_emission(10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT)
        synch.get_delta_emission(10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT)
        synch.get_delta_emission(10 * Unit("GHz"), output_unit=Unit("MJy/sr"))
        synch.get_delta_emission(10 * Unit("GHz"), output_unit=Unit("MJy/sr"))

        assert (
            synch.get_delta_emission(10 * Unit("GHz"), DEFAULT_OUTPUT_UNIT).shape
            == synch.amp.shape
        )
        assert (
            synch.get_delta_emission(10 * Unit("GHz"), output_unit="MJy/sr").shape
            == synch.amp.shape
        )


def test_diffuse_bandpass(synch_1, synch_3):
    """Test delta simulation for diffuse comps."""

    synchs = [synch_1, synch_3]

    for synch in synchs:
        synch.get_bandpass_emission([10, 11, 12] * Unit("GHz"), [0.3,0.3,0.3] * Unit("K_RJ"), DEFAULT_OUTPUT_UNIT)
        synch.get_bandpass_emission([10, 11, 12] * Unit("GHz"), [0.3,0.3,0.3] * Unit("K_RJ"), DEFAULT_OUTPUT_UNIT)
        synch.get_bandpass_emission(
            [10, 11, 12] * Unit("GHz"), [0.3,0.3,0.3] * Unit("K_RJ"), output_unit=Unit("MJy/sr")
        )
        synch.get_bandpass_emission(
            [10, 11, 12] * Unit("GHz"),
            weights=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=DEFAULT_OUTPUT_UNIT,
        )
        synch.get_bandpass_emission(
            [10, 11, 12] * Unit("GHz"),
            weights=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )
        synch.get_bandpass_emission(
            [10, 11, 12] * Unit("GHz"),
            weights=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )
        synch.get_bandpass_emission(
            [10, 11, 12] * Unit("GHz"),
            weights=[10, 11, 12] * DEFAULT_OUTPUT_UNIT,
            output_unit=Unit("MJy/sr"),
        )

        assert (
            synch.get_bandpass_emission(
                [10, 11] * Unit("GHz"), [0.3,0.3] * Unit("K_RJ"), DEFAULT_OUTPUT_UNIT
            ).shape
            == synch.amp.shape
        )
        assert (
            synch.get_bandpass_emission(
                [10, 11] * Unit("GHz"),
                weights=[11, 13] * DEFAULT_OUTPUT_UNIT,
                output_unit="MJy/sr",
            ).shape
            == synch.amp.shape
        )
