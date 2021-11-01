import pytest

from astropy.units import Quantity
import healpy as hp
import numpy as np

from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.dust import Dust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.radio import Radio

from cosmoglobe.sky.model import SkyModel


@pytest.fixture()
def synch_1():

    return Synchrotron(
        Quantity(np.random.randint(10, 20, (1, hp.nside2npix(32))), unit="K"),
        Quantity([[40]], unit="GHz"),
        beta=Quantity([[1]]),
    )


@pytest.fixture()
def synch_3():
    return Synchrotron(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="K"),
        Quantity([[40], [40], [40]], unit="GHz"),
        beta=Quantity([[1], [1], [1]]),
    )


@pytest.fixture()
def dust_1():

    return Dust(
        Quantity(np.random.randint(10, 20, (1, hp.nside2npix(32))), unit="K"),
        Quantity([[40]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 100, (1, hp.nside2npix(32)))),
        T=Quantity([[1000]], unit="K"),
    )


@pytest.fixture()
def dust_3():
    return Dust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="K"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity([[0.3], [0.4], [0.4]]),
        T=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(32))), unit="K"),
    )


@pytest.fixture()
def dust0spec():
    return Dust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="K"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity([[10], [11], [11]]),
        T=Quantity([[5], [6], [6]], unit="K"),
    )


@pytest.fixture()
def dust1spec():
    return Dust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="K"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity([[10], [11], [11]]),
        T=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(32))), unit="K"),
    )


@pytest.fixture()
def dust2spec():
    return Dust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(32))), unit="K"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(32)))),
        T=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(32))), unit="K"),
    )


@pytest.fixture()
def radio():
    amp = Quantity(np.random.randint(10, 20, (1, 12192)), unit="mJy")
    freq_ref = Quantity([[40]], unit="GHz")
    alpha = Quantity(np.random.randint(10, 20, (1, 12192)))
    radio = Radio(amp, freq_ref, alpha=alpha)
    return radio


@pytest.fixture()
def ff():
    amp = Quantity(np.random.randint(10, 20, (1, hp.nside2npix(256))), unit="uK")
    freq_ref = Quantity([[40]], unit="GHz")
    T_e = Quantity(np.random.randint(10, 20, (1, hp.nside2npix(256))), unit="K")

    return FreeFree(amp, freq_ref, T_e=T_e)


@pytest.fixture()
def ame():
    return AME(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(256))), unit="uK"),
        Quantity([[20], [50], [50]], unit="GHz"),
        freq_peak=Quantity(
            np.random.randint(10, 100, (3, hp.nside2npix(256))), unit="GHz"
        ),
    )


@pytest.fixture()
def synch():
    return Synchrotron(
        Quantity(np.random.randint(10, 20, (1, hp.nside2npix(256))), unit="uK"),
        Quantity([[40]], unit="GHz"),
        beta=Quantity([[1]]),
    )


@pytest.fixture()
def dust():
    return Dust(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(256))), unit="uK"),
        Quantity([[20], [50], [50]], unit="GHz"),
        beta=Quantity([[0.3], [0.4], [0.4]]),
        T=Quantity(np.random.randint(10, 100, (3, hp.nside2npix(256))), unit="uK"),
    )


@pytest.fixture()
def cmb():
    return CMB(
        Quantity(np.random.randint(10, 20, (3, hp.nside2npix(256))), unit="uK"),
        Quantity([[1], [1], [1]], unit="GHz"),
    )


@pytest.fixture()
def sky_model(synch, dust, ame, ff, radio, cmb):
    return SkyModel(256, {"synch":synch, "dust":dust, "ame":ame, "ff":ff, "radio":radio, "cmb":cmb})
