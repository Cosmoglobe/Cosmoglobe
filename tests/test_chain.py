import pathlib
from typing import Generator

import healpy as hp
import numpy as np
import pytest

from cosmoglobe.h5 import ChainVersion
from cosmoglobe.h5._exceptions import ChainFormatError, ChainKeyError, ChainSampleError
from cosmoglobe.h5.chain import Chain

# This path needs to exist in the test environment
CHAIN_PATH = "/Users/metinsan/Documents/doktor/Cosmoglobe_test_data/chain_test.h5"


def test_init():
    """Test initialization of a chain."""
    with pytest.raises(FileNotFoundError):
        Chain("random/path")

    with pytest.raises(ChainFormatError):
        Chain(pathlib.Path(__file__).resolve())

    assert isinstance(Chain(CHAIN_PATH), Chain)


def test_init_with_burnin():
    """Test initialization of a chain with a burn in sample."""

    with pytest.raises(ChainSampleError):
        Chain(CHAIN_PATH, burn_in=100)

    assert isinstance(Chain(CHAIN_PATH, burn_in=1), Chain)


@pytest.fixture
def chain():
    return Chain(CHAIN_PATH)


@pytest.fixture
def chain_burn_in():
    return Chain(CHAIN_PATH, burn_in=1)


def test_samples_type(chain):
    """Tests the type of the samples property."""

    assert isinstance(chain.samples, list)


def test_nsamples_type(chain):
    """Tests the type of the nsamples property."""

    assert isinstance(chain.nsamples, int)


def test_components_type(chain):
    """Tests the type of the components property."""

    assert isinstance(chain.components, list)


def test_nsamples(chain, chain_burn_in):
    """Tests the number of samples"""

    assert chain.nsamples - chain_burn_in.nsamples == 1


def test_version_type(chain):
    """Tests the type of the version property."""

    assert isinstance(chain.version, ChainVersion)


def test_path_type(chain):
    """Tests the type of the version property."""

    assert isinstance(chain.path, pathlib.Path)


def test_parameters_type(chain):
    """Tests the type of the parameters property."""

    assert isinstance(chain.parameters, dict)


def test_validate_samples(chain):
    """Tests the validate_samples decorator.

    This decorator is attached on the Chain.get() function which is decorated
    with the validate_samples decorator.
    """

    with pytest.raises(ChainSampleError):
        chain.get("dust/amp_alm", samples="000001")

    with pytest.raises(ChainSampleError):
        chain.get("dust/amp_alm", samples=100)

    with pytest.raises(ChainSampleError):
        chain.get("dust/amp_alm", samples=range(0, 200))

    with pytest.raises(ChainSampleError):
        chain.mean("dust/amp_alm", samples=100)

    with pytest.raises(ChainSampleError):
        chain.load("dust/amp_alm", samples=100)

    chain.get("dust/amp_alm", samples=1)
    chain.mean("dust/amp_alm", samples=range(2))
    chain.mean("dust/amp_alm", samples=[0,1])
    chain.load("dust/amp_alm", samples=-1)


def test_unpack_alms(chain):
    """Tests the unpack_alms decorator.

    This decorator is attached on the Chain.get() function which is decorated
    with the validate_samples decorator.
    """

    assert chain.get("dust/amp_alm", samples=range(2)).ndim == 3
    assert chain.get("dust/amp_alm", samples=-1).ndim == 2
    assert isinstance(chain.get("dust/amp_lmax", samples=-1), np.int32)


def test_validate_key(chain):
    """Tests the validate_key decorator.

    This decorator is attached on the Chain.get() function which is decorated
    with the validate_keydecorator.
    """

    with pytest.raises(ChainKeyError):
        chain.get("dust/invalid_key")

    with pytest.raises(ChainKeyError):
        chain.mean("dust/invalid_key")

    with pytest.raises(ChainKeyError):
        chain.load("dust/invalid_key")

    with pytest.raises(ChainKeyError):
        chain["000010/dust/invalid_key"]


def test_get(chain):
    """Tests the return type of Chain.get()."""

    with pytest.raises(ChainKeyError):
        chain.get("random/path")

    with pytest.raises(ChainSampleError):
        chain.get("dust/amp_lmax", samples=range(1000))

    assert isinstance(chain.get("dust/amp_lmax"), np.ndarray)
    assert isinstance(chain.get("dust/amp_lmax", samples=-1), np.int32)
    assert isinstance((alms := chain.get("dust/amp_alm", samples=-1)), np.ndarray)
    hp.alm2map(alms, 32)


def test_mean(chain):
    """Tests the return type of Chain.mean()."""

    with pytest.raises(ChainKeyError):
        chain.mean("random/path")

    with pytest.raises(ChainSampleError):
        chain.mean("dust/amp_lmax", samples=range(1000))

    assert isinstance(chain.mean("dust/amp_lmax"), np.int32)
    assert isinstance(chain.mean("dust/amp_lmax", samples=-1), np.int32)
    assert isinstance((alms := chain.mean("dust/amp_alm")), np.ndarray)
    hp.alm2map(alms, 32)


def test_load(chain):
    """Tests the return type of Chain.load()."""

    with pytest.raises(ChainKeyError):
        chain.load("random/path")

    with pytest.raises(ChainSampleError):
        chain.load("dust/amp_lmax", samples=range(1000))

    assert isinstance(chain.load("dust/amp_lmax"), Generator)
    for value in chain.load("dust/amp_lmax", samples=-1):
        assert isinstance(value, np.int32)

    for value in chain.load("dust/amp_alm", samples=-1):
        assert isinstance(value, np.ndarray)
        hp.alm2map(value, 32)


def test_getitem(chain):
    """Tests the __getitem__ function of chain."""

    with pytest.raises(ChainKeyError):
        chain["random/path"]

    assert isinstance(chain["parameters/dust/class"], str)
    assert isinstance(chain["000001/dust/amp_alm"], np.ndarray)
    assert isinstance(chain["000001/dust/amp_lmax"], np.int32)

    hp.alm2map(chain["000001/dust/amp_alm"], 32)


def test_format_samples(chain):
    """Tests the _format_samples function of chain."""

    assert chain._format_samples([0, 1]) == ["000000", "000001"]
