from enum import Enum, auto
from pathlib import Path
import textwrap
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import h5py
from numba import njit
import numpy as np

from cosmoglobe.h5.exceptions import ChainFormatError, ChainKeyError, ChainSampleError
from cosmoglobe.h5.decorators import validate_key, validate_samples
from cosmoglobe.sky import COSMOGLOBE_COMPS


PARAMETER_GROUP_NAME = "parameters"


class ChainVersion(Enum):
    """The version number of the chain."""

    OLD = auto()
    NEW = auto()


class Chain:
    """An interface for Cosmoglobe chainfiles."""

    def __init__(self, path: Union[str, Path], burn_in: Optional[int] = None) -> None:
        """Validate and initialize the Chain object."""

        if not (path := Path(path)).is_file():
            raise FileNotFoundError(f"{path.name} was not found")
        try:
            with h5py.File(path, "r") as file:
                pass
        except OSError:
            raise ChainFormatError(f"{path.name} is not a HDF5 file")

        with h5py.File(path, "r") as file:
            samples = list(file.keys())
            if not samples:
                raise ChainFormatError("chain has no samples")
            try:
                samples.remove(PARAMETER_GROUP_NAME)
                version = ChainVersion.NEW
            except ValueError:
                version = ChainVersion.OLD

            sampled_groups = list(file[samples[0]].keys())
            components = [
                group for group in sampled_groups if group in COSMOGLOBE_COMPS
            ]

            if version is ChainVersion.NEW:
                parameters = {}
                for component, group in file[PARAMETER_GROUP_NAME].items():
                    parameters[component] = {}
                    for key, value in group.items():
                        if np.issubdtype(value.dtype, np.string_):
                            value = value.asstr()
                        parameters[component][key] = value[()]
            else:
                parameters = None

        if burn_in is None:
            self._samples = samples
        else:
            if burn_in >= (nsamples := len(samples)):
                raise ChainSampleError(f"{burn_in=} out of range with {nsamples=}")
            self._samples = samples[burn_in:]
        self._components = components
        self._parameters = parameters
        self._path = path
        self._version = version

    @property
    def samples(self) -> List[str]:
        """List of all samples in the chain."""

        return self._samples

    @property
    def nsamples(self) -> int:
        """Number of samples in the chain."""

        return len(self.samples)

    @property
    def components(self) -> List[str]:
        """List of the sky components in the chain."""

        return self._components

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary of the parameters in the parameter group in the chain."""

        return self._parameters

    @property
    def path(self) -> Path:
        """Path to the chainfile."""

        return self._path

    @property
    def version(self) -> ChainVersion:
        """Chain version."""

        return self._version

    @validate_key
    @validate_samples
    def get(
        self,
        key: str,
        *,
        samples: Optional[Union[range, int]] = None,
    ) -> List[Any]:
        """Returns the value of an key for all samples.

        Parameters
        ----------
        key
            The path to an item that has been sampled in the chain, e.g
            'dust/amp_alm'.
        samples
            An int or a range of samples for which to return the value. If
            None, all samples in the chain are used.

        Returns
        -------
        values
            The value of the key for each samples.
        """

        with h5py.File(self.path, "r") as file:
            values = [file[f"{sample}/{key}"][()] for sample in samples]

        if "alm" in key:
            values = self._unpack_alms(key, values)

        return np.asarray(values) if len(values) != 1 else values[0]

    @validate_key
    @validate_samples
    def mean(
        self,
        key: str,
        *,
        samples: Optional[Union[range, int]] = None,
    ) -> Any:
        """Returns the mean of an key over all samples.

        Parameters
        ----------
        key
            The path to an item that has been sampled in the chain, e.g
            'dust/amp_alm'.
        samples
            An int or a range of samples to average over. If None, all
            samples in the chain are used.

        Returns
        -------
        value
            The averaged value of the key over all samples.
        """

        with h5py.File(self.path, "r") as file:
            value = file[f"{samples[0]}/{key}"][()]
            dtype = value.dtype.type
            if len(samples) > 1:
                for sample in samples[1:]:
                    value += file[f"{sample}/{key}"][()]

        value = dtype(value / len(samples))

        if "alm" in key:
            value = self._unpack_alms(key, value)

        return value

    @validate_key
    @validate_samples
    def load(
        self,
        key: str,
        *,
        samples: Optional[Union[int, range]] = None,
    ) -> Generator:
        """Returns a generator to be used in a for loop.

        Parameters
        ----------
        key
            The path to an item that has been sampled in the chain, e.g
            'dust/amp_alm'.
        samples
            An int or a range of samples to average over. If None, all
            samples in the chain are used.

        Returns
        -------
            A generator that can be looped over to yield each sampled value.
        """

        with h5py.File(self.path, "r") as file:
            for sample in samples:
                value = file[f"{sample}/{key}"][()]
                if "alm" in key:
                    value = self._unpack_alms(key, value)

                yield value

    @validate_key
    def __getitem__(self, key: str) -> Any:
        """Returns the value of a key from the chain.

        NOTE: alms are not unpacked into HEALPIX convention. Use either
        the `get`, `mean` or `load` functions to retrieve unpacked alms.

        Parameters
        ----------
        key
            The *full* path to an item in the chain.

        Returns
        -------
            The value of the key.
        """

        with h5py.File(self.path, "r") as file:
            item = file[key]
            if isinstance(item, h5py.Group):
                return list(item.keys())

            elif isinstance(item, h5py.Dataset):
                if np.issubdtype(item.dtype, np.string_):
                    return item.asstr()[()]
                return item[()]

    @staticmethod
    def _to_chain_sample_format(
        samples: Union[List[int], int]
    ) -> Union[List[str], str]:
        """Converts sample numbers to the format in the chain files."""

        if isinstance(samples, Iterable):
            return [f"{sample:06d}" for sample in list(samples)]

        return f"{samples:06d}"

    def _unpack_alms(
        self, key: str, values: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Unpacks alms from Commander to HEALPIX format."""

        try:
            lmax = self[f"{self.samples[0]}/{key[:-3]}lmax"]
        except KeyError:
            raise ChainKeyError(
                f"{key} does not exist. Cannot unpack alms from chain "
                "without an lmax present in the chain"
            )
        if isinstance(values, list):
            return [unpack_alms_from_chain(value, lmax) for value in values]

        return unpack_alms_from_chain(values, lmax)

    def __str__(self) -> str:
        """Representation of the chain."""

        COL_LEN = 40
        META = {
            "Num Samples": self.nsamples,
            "Num Components": len(self.components),
            "Size": f"{self.path.stat().st_size / (1024 * 1024 * 1024):.3f}" + " GB",
        }

        def center(string: str, fill=" ") -> str:
            white_space_len = (COL_LEN - len(string)) // 2
            white_space = fill * white_space_len
            output = f"{white_space}{string}{white_space}"
            if len(output) < COL_LEN:
                output += fill
            return output

        main_repr = "\n"
        main_repr += "-" * COL_LEN + "\n"
        main_repr += center(self.path.name) + "\n"
        main_repr += "-" * COL_LEN + "\n"
        main_repr += "\n"

        for key, value in META.items():
            main_repr += f"{key:<{COL_LEN//2 - 1}}{'='}{value:>{COL_LEN//2}}\n"

        main_repr += "\n"
        main_repr += center(" Components ", fill="-") + "\n"
        main_repr += (
            center(textwrap.fill("  ".join(self.components), width=COL_LEN)) + "\n"
        )
        main_repr += "\n"
        main_repr += "-" * COL_LEN + "\n"

        return main_repr


@njit
def unpack_alms_from_chain(data, lmax):
    """Unpacks alms from the Commander chain output.

    Unpacking algorithm:
    https://github.com/trygvels/c3pp/blob/2a2937926c260cbce15e6d6d6e0e9d23b0be1262/src/tools.py#L9

    Parameters
    ----------
    data
        alms from a commander chainfile.
    lmax
        Maximum value for l used in the alms.

    Returns
    -------
    alms
        Unpacked version of the Commander alms (2-dimensional array)
    """

    n = len(data)
    n_alms = int(lmax * (2 * lmax + 1 - lmax) / 2 + lmax + 1)
    alms = np.zeros((n, n_alms), dtype=np.complex128)

    for sigma in range(n):
        i = 0
        for l in range(lmax + 1):
            j_real = l ** 2 + l
            alms[sigma, i] = complex(data[sigma, j_real], 0.0)
            i += 1

        for m in range(1, lmax + 1):
            for l in range(m, lmax + 1):
                j_real = l ** 2 + l + m
                j_comp = l ** 2 + l - m
                alms[sigma, i] = (
                    complex(
                        data[sigma, j_real],
                        data[sigma, j_comp],
                    )
                    / np.sqrt(2.0)
                )
                i += 1

    return alms
