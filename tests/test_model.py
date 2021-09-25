import pytest

from astropy.units import Unit, Quantity
import numpy as np
import healpy as hp

from cosmoglobe.sky import DEFAULT_OUTPUT_UNIT
from cosmoglobe.sky.components import Synchrotron, Dust, AME, Radio
from cosmoglobe.sky._exceptions import NsideError, SkyModelComponentError
from cosmoglobe.sky.sky_model import SkyModel


def test_init_sky_model_nside(synch_3, dust_3):
    sky_model = SkyModel(32, [synch_3, dust_3])
    assert isinstance(sky_model.nside, int)
    assert isinstance(sky_model.components, dict)

    with pytest.raises(NsideError):
        SkyModel(66, [synch_3])

    with pytest.raises(TypeError):
        SkyModel("sdf", [synch_3])

    with pytest.raises(SkyModelComponentError):
        SkyModel(32, [1])


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


def test_synch():
    """Tests a sim of synch."""

    synch = Synchrotron(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128)))),
    )
    sky_model = SkyModel(128, [synch])
    assert sky_model(100*Unit("GHz")).shape == synch.amp.shape
    assert sky_model([100,101,102]*Unit("GHz")).shape == synch.amp.shape
    assert sky_model([100,101]*Unit("GHz"), bandpass=[3,5]*Unit("uK")).shape == synch.amp.shape

def test_dust():
    """Tests a sim of dust."""

    dust = Dust(
        Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))), unit="uK"),
        Quantity([[40], [50], [50]], unit="GHz"),
        beta=Quantity([[1],[2],[2]]),
        T=Quantity(np.random.randint(10, 30, (3, hp.nside2npix(128))),unit="K"),
    )
    sky_model = SkyModel(128, [dust])
    assert sky_model(100*Unit("GHz")).shape == dust.amp.shape
    assert sky_model([100,101,102]*Unit("GHz")).shape == dust.amp.shape
    assert sky_model([100,101]*Unit("GHz"), bandpass=[3,5]*Unit("uK")).shape == dust.amp.shape

def test_radio():
    """Tests a sim of radio."""

    radio = Radio(
        Quantity(np.random.randint(10, 30, (1, 12192)), unit="mJy"),
        Quantity([[40]], unit="GHz"),
        alpha=Quantity(np.random.randint(10, 30, (1, 12192))),
    )
    sky_model = SkyModel(128, [radio])
    assert sky_model(100*Unit("GHz")).shape == (1, hp.nside2npix(128))
    assert sky_model([100,101,102]*Unit("GHz")).shape == (1, hp.nside2npix(128))
    assert sky_model([100,101]*Unit("GHz"), bandpass=[3,5]*Unit("uK")).shape == (1, hp.nside2npix(128))
