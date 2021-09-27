import pytest

from astropy.units import Unit, Quantity
import numpy as np
import healpy as hp

from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky.components import Synchrotron, Dust, AME, Radio
from cosmoglobe.sky._exceptions import NsideError, ComponentError, ComponentNotFoundError
from cosmoglobe.sky.model import SkyModel


def test_init_sky_model_nside(synch_3, dust_3):
    sky_model = SkyModel(32, [synch_3, dust_3])
    assert isinstance(sky_model.nside, int)
    assert isinstance(sky_model.components, dict)

    with pytest.raises(NsideError):
        SkyModel(66, [synch_3])

    with pytest.raises(TypeError):
        SkyModel("sdf", [synch_3])

    with pytest.raises(ComponentError):
        SkyModel(32, [1])

    synch = Synchrotron(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32)))),
    )
    dust = Dust(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity([[1], [2], [2]]),
        T=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="K"),
    )

    with pytest.raises(NsideError):
        sky_model = SkyModel(32, [synch, dust])


def test_comp_arg(sky_model):
    """Tests that all comps in the comp arg is present."""
    synch = Synchrotron(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32)))),
    )
    dust = Dust(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity([[1], [2], [2]]),
        T=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(32))), unit="K"),
    )
    sky_model = SkyModel(32, [synch, dust])
    sky_model(100 * Unit("GHz"), fwhm=80 * Unit("arcmin"), components=["synch"])

    with pytest.raises(ValueError):
        sky_model(
            100 * Unit("GHz"),
            fwhm=80 * Unit("arcmin"),
            components=["synch", "dust", "radio"],
        )


def test_repr(sky_model):
    """Test that the repr is printed."""

    print(sky_model)


def test_iter_model(sky_model):
    """Tests the iter method."""

    for component in sky_model:
        component


def test_simulate(sky_model):
    """Test calling the model."""

    emission = sky_model(50 * Unit("GHz"))
    assert emission.unit == DEFAULT_OUTPUT_UNIT
    assert emission.shape == sky_model.components["dust"].amp.shape

    emission = sky_model(np.arange(1, 10) * Unit("GHz"))
    assert emission.unit == DEFAULT_OUTPUT_UNIT
    assert emission.shape == sky_model.components["dust"].amp.shape

    emission = sky_model(
        np.arange(1, 10) * Unit("GHz"), bandpass=np.arange(1, 10) * Unit("uK")
    )
    assert emission.unit == DEFAULT_OUTPUT_UNIT
    assert emission.shape == sky_model.components["dust"].amp.shape

    emission = sky_model(
        np.arange(1, 10) * Unit("GHz"),
        bandpass=np.arange(1, 10) * Unit("uK"),
        output_unit="MJy/sr",
    )
    assert emission.unit == Unit("MJy/sr")

    emission = sky_model(
        np.arange(1, 10) * Unit("GHz"),
        bandpass=np.arange(1, 10) * Unit("uK"),
        output_unit="uK_RJ",
    )
    assert emission.unit == Unit("uK")

def test_synch():
    """Tests a sim of synch."""

    synch = Synchrotron(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128)))),
    )
    sky_model = SkyModel(128, [synch])
    assert sky_model(100 * Unit("GHz")).shape == synch.amp.shape
    assert sky_model([100, 101, 102] * Unit("GHz")).shape == synch.amp.shape
    assert (
        sky_model([100, 101] * Unit("GHz"), bandpass=[3, 5] * Unit("uK")).shape
        == synch.amp.shape
    )


def test_dust():
    """Tests a sim of dust."""

    dust = Dust(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity([[1], [2], [2]]),
        T=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="K"),
    )
    sky_model = SkyModel(128, [dust])
    assert sky_model(100 * Unit("GHz")).shape == dust.amp.shape
    assert sky_model([100, 101, 102] * Unit("GHz")).shape == dust.amp.shape
    assert (
        sky_model([100, 101] * Unit("GHz"), bandpass=[3, 5] * Unit("uK")).shape
        == dust.amp.shape
    )


def test_radio():
    """Tests a sim of radio."""

    radio = Radio(
        Quantity(np.random.randint(10, 30, (1, 12192)), unit="mJy"),
        Quantity([[40]], unit="GHz"),
        alpha=Quantity(np.random.randint(10, 30, (1, 12192))),
    )
    sky_model = SkyModel(256, [radio])
    assert sky_model(100 * Unit("GHz"), fwhm=60 * Unit("arcmin")).shape == (
        1,
        hp.nside2npix(256),
    )
    assert sky_model([100, 101, 102] * Unit("GHz")).shape == (1, hp.nside2npix(256))
    assert sky_model(
        [100, 101] * Unit("GHz"), bandpass=[3, 5] * Unit("uK"), fwhm=60 * Unit("arcmin")
    ).shape == (1, hp.nside2npix(256))

    with pytest.raises(ValueError):
        sky_model(100 * Unit("GHz"), fwhm=10 * Unit("arcmin")).shape == (
            1,
            hp.nside2npix(256),
        )


def test_ame():
    """Tests a sim of ame."""

    ame = AME(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [30], [30]], unit="GHz"),
        freq_peak=Quantity([[100], [100], [100]], unit="GHz"),
    )
    sky_model = SkyModel(128, [ame])
    assert sky_model(100 * Unit("GHz")).shape == (3, hp.nside2npix(128))
    assert (sky_model(0.001 * Unit("GHz")) == 0).all()
    assert sky_model([100, 101, 102] * Unit("GHz")).shape == (3, hp.nside2npix(128))
    assert sky_model([100, 101] * Unit("GHz"), bandpass=[3, 5] * Unit("uK")).shape == (
        3,
        hp.nside2npix(128),
    )


def test_remove_dipole(sky_model):
    """Tests the remove_dipole function."""

    sky_model.remove_dipole()

    ame = AME(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [30], [30]], unit="GHz"),
        freq_peak=Quantity([[100], [100], [100]], unit="GHz"),
    )
    model = SkyModel(128, [ame])
    with pytest.raises(ComponentNotFoundError):
        model.remove_dipole()
