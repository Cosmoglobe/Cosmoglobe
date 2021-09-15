from pathlib import Path
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Union

import h5py
from numba import njit
import numpy as np

from cosmoglobe.sky import COSMOGLOBE_COMPS
from cosmoglobe.h5.exceptions import (
    ChainFormatError,
    ChainItemNotFoundError,
    ChainSampleError,
)


PARAMETER_GROUP_NAME = "parameters"


class Chain:
    """An interface for Cosmoglobe chainfiles."""

    def __init__(self, path: Union[str, Path], burn_in: Optional[int] = None) -> None:
        """Validate and initialize the Chain object."""

        if not (path := Path(path)).is_file():
            raise FileNotFoundError(f"{path.name} was not found")
        try:
            with h5py.File(path, "r") as file:
                ...
        except OSError:
            raise ChainFormatError(f"{path.name} is not a HDF5 file")

        with h5py.File(path, "r") as file:
            samples = list(file.keys())
            if not samples:
                raise ChainFormatError
            try:
                samples.remove(PARAMETER_GROUP_NAME)
            except ValueError:
                raise ChainFormatError
            sampled_groups = list(file[samples[0]].keys())
            components = [
                group for group in sampled_groups if group in COSMOGLOBE_COMPS
            ]

            parameters = {}
            for component, group in file[PARAMETER_GROUP_NAME].items():
                parameters[component] = {}
                for key, value in group.items():
                    if np.issubdtype(value.dtype, np.string_):
                        value = value.asstr()
                    parameters[component][key] = value[()]

        self.burn_in = burn_in
        self._path = path
        self._samples = samples
        self._components = components
        self._parameters = parameters

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

    def get(
        self,
        item: str,
        samples: Optional[Iterable[str]] = None,
        burn_in: Optional[int] = None,
    ) -> List[Any]:
        """Returns the value of an item for all samples.

        Parameters
        ----------
        item
            A sampled item.
        samples
            A list of samples for which to return the item value. If None,
            all samples are selected. Defaults to None.
        burn_in
            The burn_in sample. If provided, all samples before the burn_in
            is ignored. If None, but the chain was initialized with a burn_in,
            that burn_in value will be used. Defaults to None.

        Returns
        -------
        values
            The value of the item for each samples.
        """

        samples = self._process_samples(item, samples, burn_in)

        with h5py.File(self.path, "r") as file:
            try:
                values = [file[f"{sample}/{item}"][()] for sample in samples]
            except KeyError:
                raise ChainItemNotFoundError(
                    f"{item=!r} was not found. The item must be a dataset "
                    "present in all sample groups"
                )

        # If the extracted value is an alm, we unpack it from the Commander
        # convention into the HEALPIX convention.
        if "alm" in item:
            try:
                lmax = self[f"{self.samples[0]}/{item[:-3]}lmax"]
            except KeyError:
                raise ChainItemNotFoundError(
                    f"{item} does not exist. Cannot unpack alms from chain "
                    "without an lmax"
                )
            values = [unpack_alms_from_chain(value, lmax) for value in values]

        if len(values) == 1:
            return values[0]

        return np.asarray(values)

    def mean(
        self,
        item: str,
        samples: Optional[Iterable[str]] = None,
        burn_in: Optional[int] = None,
    ) -> Any:
        """Returns the mean of an item over all samples.

        Parameters
        ----------
        item
            A sampled item.
        samples
            A list of samples to average over. If None, all samples are
            averaged. Defaults to None.
        burn_in
            The burn_in sample. If provided, all samples before the burn_in
            is ignored. If None, but the chain was initialized with a burn_in,
            that burn_in value will be used. Defaults to None.

        Returns
        -------
        value
            The averaged value of the item over all samples.
        """

        samples = self._process_samples(item, samples, burn_in)

        with h5py.File(self.path, "r") as file:
            try:
                value = file[f"{samples[0]}/{item}"][()]
            except KeyError:
                raise ChainItemNotFoundError(
                    f"{item=!r} was not found. The item must be a dataset "
                    "present in all sample groups"
                )
            dtype = value.dtype.type
            if len(samples) > 1:
                for sample in samples[1:]:
                    value += file[f"{sample}/{item}"][()]

        value = dtype(value / len(samples))

        # If the extracted value is an alm, we unpack it from the Commander
        # convention into the HEALPIX convention.
        if "alm" in item:
            try:
                lmax = self[f"{self.samples[0]}/{item[:-3]}lmax"]
            except KeyError:
                raise ChainItemNotFoundError(
                    f"{item} does not exist. Cannot unpack alms from chain "
                    "without an lmax"
                )
            value = unpack_alms_from_chain(value, lmax)

        return value

    def __getitem__(self, key) -> Any:
        """Returns an item from the chain.

        Note that alms are not unpacked into HEALPIX convention using the
        key lookup."""

        with h5py.File(self.path, "r") as file:
            try:
                item = file[key]
            except KeyError as error:
                raise ChainItemNotFoundError(error)

            if isinstance(item, h5py.Group):
                return list(item.keys())

            elif isinstance(item, h5py.Dataset):
                if np.issubdtype(item.dtype, np.string_):
                    return item.asstr()[()]
                return item[()]

    def _process_samples(self, item: str, samples: list, burn_in: int) -> list:
        """Validates and process inputted samples."""

        if item.startswith(PARAMETER_GROUP_NAME):
            raise ValueError(
                "Can only extract items from sample groups. To access "
                f"items in the {PARAMETER_GROUP_NAME} group, use dictionary "
                "lookup instead (chain['parameters/...'])"
            )

        if samples is None or samples == "all":
            samples = self.samples
        elif samples == -1:
            samples = [self.samples[-1]]
        elif isinstance(samples, int):
            samples = self._int_to_sample([samples])
            if not samples[0] in self.samples:
                raise ChainSampleError(f"input sample {samples} is not in the chain")
        elif isinstance(samples, Iterable):
            samples = list(samples)
        else:
            raise ChainSampleError(
                "input samples must be an iterable or an int pointing to a sample"
            )
        if samples[0] == 0:
            raise ChainSampleError("Chain samples starts at 1, not 0")
        if len(samples) > self.nsamples:
            raise ChainSampleError(f"Chain only has {self.nsamples} samples")

        if burn_in is None:
            burn_in = self.burn_in
        if burn_in is not None:
            if burn_in >= self.nsamples:
                raise ValueError(f"{burn_in=} out of range for {self.nsamples=}.")
            else:
                samples = samples[burn_in:]

        if all(isinstance(sample, int) for sample in samples):
            samples = self._int_to_sample(samples)

        return samples

    @staticmethod
    def _int_to_sample(samples: Union[List[int], int]) -> Union[List[str], str]:
        return [f"{sample:06d}" for sample in list(samples)]

    def __str__(self) -> str:
        """Representation of the chain."""

        COL_LEN = 40

        stats = {
            "Num samples": self.nsamples,
            "Num components": len(self.components),
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

        for key, value in stats.items():
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
def unpack_alms_from_chain(data: np.ndarray, lmax: int) -> np.ndarray:
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
