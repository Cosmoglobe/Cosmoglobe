from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

import h5py
import healpy as hp
import numpy as np

from cosmoglobe.h5 import PARAMETER_GROUP_NAME, ChainVersion
from cosmoglobe.h5._alms import unpack_alms as unpack_alms_
from cosmoglobe.h5._alms import unpack_alms_from_chain
from cosmoglobe.h5._decorators import unpack_alms, validate_key, validate_samples
from cosmoglobe.h5._exceptions import ChainFormatError, ChainKeyError, ChainSampleError
from cosmoglobe.sky.components._labels import SkyComponentLabel


class Chain:
    """An interface for Cosmoglobe chainfiles.

    This class aims to provide a convenient interface for working with
    Cosmoglobe chain files.
    """

    def __init__(self, path: str | Path, burn_in: Optional[int] = None) -> None:
        """Validate and initialize the Chain object.

        Parameters
        ----------
        path
            Path to the chainfile.
        burn_in
            Burn in sample. All samples prior to (and including) the burn
            in sample is discarded.
        """

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
                group
                for group in sampled_groups
                if group in [label.value for label in SkyComponentLabel]
            ]

            parameters: dict[str, dict[str, Any]] = {}
            if version is ChainVersion.NEW:
                for component, group in file[PARAMETER_GROUP_NAME].items():
                    parameters[component] = {}
                    for key, value in group.items():
                        if np.issubdtype(value.dtype, np.string_):
                            value = value.asstr()
                        parameters[component][key] = value[()]

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
    def samples(self) -> list[str]:
        """List of all samples in the chain."""

        return self._samples

    @property
    def nsamples(self) -> int:
        """Number of samples in the chain."""

        return len(self.samples)

    @property
    def components(self) -> list[str]:
        """List of the sky components in the chain."""

        return self._components

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
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

    @property
    def tree(self):
        """Prints group and dataset structure of the chainfile."""

        def print_attrs(name, _):
            space = name.count("/") * "    "
            path = name.split("/")
            if path[0] not in ("000000", "parameters"):
                return
            item_name = path[-1]
            print(space + item_name)

        with h5py.File(self.path, "r") as file:
            file.visititems(print_attrs)

    @validate_key
    @validate_samples
    @unpack_alms
    def get(
        self,
        key: str,
        *,
        samples: Optional[range | int | Sequence[int]] = None,
        unpack: bool = True,
    ) -> Any:
        """Returns the value of an key for all samples.

        Parameters
        ----------
        key
            The path to an item that has been sampled in the chain, e.g
            'dust/amp_alm'.
        samples
            An int or a range of samples for which to return the value. If
            None, all samples in the chain are used.
        unpack
            If True, alms are unpacked from Commander format to healpy format.
            Default is True

        Returns
        -------
        values
            The value of the key for each samples.
        """

        with h5py.File(self.path, "r") as file:
            values = [file[f"{sample}/{key}"][()] for sample in samples]

        return np.asarray(values) if len(values) != 1 else values[0]

    @validate_key
    @validate_samples
    @unpack_alms
    def mean(
        self,
        key: str,
        *,
        samples: Optional[range | int | Sequence[int]] = None,
        unpack: bool = True,
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
        unpack
            If True, alms are unpacked from Commander format to healpy format.
            Default is True
            
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
        return dtype(value / len(samples))  # Converting back to original dtype

    @validate_key
    @validate_samples
    def stddev(
        self,
        key: str,
        *,
        alm2map=False,
        samples: Optional[range | int | Sequence[int]] = None,
    ) -> Any:
        """Returns the stddev of an key over all samples.

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
        mu = dtype(value / len(samples))  # Converting back to original dtype

        # Calculate in map-space if alm2map
        if alm2map:
            comp, quantity = key.split("/")
            nside = self.parameters[comp]["nside"]
            pol = True if quantity.startswith("amp") else False
            fwhm = self.parameters[comp]["fwhm"]
            lmax = 3*nside
            mu = hp.alm2map(unpack_alms_(mu, lmax), nside=nside, lmax=lmax, fwhm=fwhm, pol=pol, pixwin=True,)


        with h5py.File(self.path, "r") as file: 
            x = file[f"{samples[0]}/{key}"][()] 
            if alm2map: x = hp.alm2map(unpack_alms_(x, lmax), nside=nside, lmax=lmax, fwhm=fwhm, pol=pol, pixwin=True,)
            dtype = x.dtype.type
            numerator = (x - mu)**2
            if len(samples) > 1:
                for sample in samples[1:]:
                    x = file[f"{sample}/{key}"][()]
                    if alm2map: x = hp.alm2map(unpack_alms_(x, lmax), nside=nside, lmax=lmax, fwhm=fwhm, pol=pol, pixwin=True,)
                    numerator += (x - mu)**2

        return dtype(np.sqrt(numerator/len(samples)))  # Converting back to original dtype

    @validate_key
    @validate_samples
    def load(
        self,
        key: str,
        *,
        samples: Optional[range | int | Sequence[int]] = None,
    ) -> Generator:
        """Returns a generator to be used in a for loop.

        NOTE to devs: The unpack_alms decorator wont work on this function
        due to it not processing the returned data until it is iterated over.

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
                    value = unpack_alms_from_chain(self, value, key)

                yield value

    @validate_key
    @unpack_alms
    def __getitem__(self, key: str, *, unpack: bool = True) -> Any:
        """Returns the value of a key from the chain.

        Parameters
        ----------
        key
            The *full* path to an item in the chain.
        unpack
            If True, alms are unpacked from Commander format to healpy format.
            Default is True

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

    def _format_samples(self, samples: list[int] | int) -> list[str] | str:
        """Converts a range to the string format of the samples in the chain."""

        leading_zeros = len(self.samples[0])
        if isinstance(samples, list):
            return [f"{sample:0{leading_zeros}d}" for sample in samples]

        return f"{samples:0{leading_zeros}d}"

    def __str__(self) -> str:
        """Representation of the chain."""

        COL_LEN = 40
        CHAIN_META = {
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

        if ".astropy/cache" in str(self.path):
            name = "cached chainfile"
        else:
            name = self.path.name

        main_repr = "\n"
        main_repr += "-" * COL_LEN + "\n"
        main_repr += center(name) + "\n"
        main_repr += "-" * COL_LEN + "\n"
        main_repr += "\n"

        for key, value in CHAIN_META.items():
            main_repr += f"{key:<{COL_LEN//2 - 1}}{'='}{value:>{COL_LEN//2}}\n"

        main_repr += "\n"
        main_repr += center(" Components ", fill="-") + "\n"
        main_repr += (
            center(textwrap.fill("  ".join(self.components), width=COL_LEN)) + "\n"
        )
        main_repr += "\n"
        main_repr += "-" * COL_LEN + "\n"

        return main_repr

    @validate_samples
    def copy(
        self,
        samples: int | Sequence[int] | range = -1,
        new_name: Optional[str] = None,
    ) -> None:
        """Creates a copy of the chain with a single or multiple samples."""

        if new_name is None:
            new_name = self.path.stem + "_copy.h5"

        with h5py.File(new_name, "x") as new_chain:
            with h5py.File(self.path, "r") as chain:
                for idx, sample in enumerate(samples):
                    group = chain[sample]
                    chain.copy(
                        source=group, dest=new_chain, name=self._format_samples(idx)
                    )

                parameter_group = chain["parameters"]
                chain.copy(
                    source=parameter_group, dest=new_chain, name=parameter_group.name
                )

    def combine(
        self,
        other_chain: Chain,
        group_list: Sequence[str],
        new_name: Optional[str] = None,
    ) -> None:
        """Creates a new chainfile that combines specific groups from two chains."""

        sample = other_chain.samples[0]
        for group in group_list:
            try:
                other_chain[f"{sample}/{group}"]
            except ChainKeyError:
                raise ChainKeyError(f"group {group} does not exist in `other chain`.")

        if new_name is None:
            new_name = self.path.stem + "_combined.h5"

        self.copy(samples=-1, new_name=new_name)

        with h5py.File(new_name, "r+") as new_chain:
            with h5py.File(other_chain.path, "r") as chain:
                sample = other_chain.samples[0]
                for group in group_list:
                    if group in new_chain[sample].keys():
                        del new_chain[f"{sample}/{group}"]
                    group_to_copy = chain[f"{sample}/{group}"]
                    chain.copy(
                        source=group_to_copy, dest=new_chain, name=group_to_copy.name
                    )

                    if group in chain[f"parameters"].keys():
                        del new_chain[f"parameters/{group}"]
                        param_to_copy = chain[f"parameters/{group}"]
                        chain.copy(
                            source=param_to_copy,
                            dest=new_chain,
                            name=param_to_copy.name,
                        )


def copy_chain(
    chain: str | Path | Chain,
    samples: int | Sequence[int] | range = -1,
    new_name: Optional[str] = None,
) -> None:
    """Creates a copy of the chain with a single or multiple samples.

    Parameters
    ----------
    chain
        Path to the chain file or the `Chain` object to copy.
    samples
        Samples to copy. Can be an int, a list of ints or a python range object.
    new_name
        Name of the chain copy. If None, a default is "{chain.name}_copy.h5"
    """

    if not isinstance(chain, Chain):
        chain = Chain(chain)

    chain.copy(samples=samples, new_name=new_name)


def combine_chains(
    chain: str | Path | Chain,
    other_chain: str | Path | Chain,
    group_list: Sequence[str],
    new_name: Optional[str] = None,
) -> None:
    """Creates a new chainfile that combines specific groups from two chains.

    The new file will contain all content from `chain`, except for the content
    within the groups in the `group_list`, which are taken from `other_chain`
    instead.

    Parameters
    ----------
    chain
        Path to chain file. This chain defines all the chain whos content you
        want to overwrite in a new combined file.
    other_chain
        Path to chain file.  This chain contains the groups you want to
        overwrite in `chain`.
    group_list
        List of hdf5 groups that will be overwritten in the new chainfile.
    new_name
        Name of the chain copy. If None, a default is "{chain.name}_copy.h5"
    """

    if not isinstance(chain, Chain):
        chain = Chain(chain)
    if not isinstance(other_chain, Chain):
        other_chain = Chain(other_chain)

    chain.combine(other_chain, group_list=group_list, new_name=new_name)
