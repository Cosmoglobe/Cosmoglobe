from functools import wraps
from typing import Callable

import h5py

from cosmoglobe.h5.exceptions import ChainKeyError, ChainSampleError


def validate_key(func: Callable) -> Callable:
    """Decotrator to check if the requested key exists in the chain."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        chain, key, *_ = args

        if key.split("/")[0] not in chain.samples:
            path = f"{chain.samples[0]}/{key}"
        else:
            path = key
        with h5py.File(chain.path, "r") as file:
            try:
                file[path]
            except KeyError:
                raise ChainKeyError(f"{key=} not found in chain")
        return func(*args, **kwargs)

    return wrapper


def validate_samples(func: Callable) -> Callable:
    """Decotrator to check if the samples exist."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        chain, *_ = args
        samples = kwargs.get("samples")
        if samples is None:
            samples = chain.samples
        elif isinstance(samples, int):
            try:
                samples = [chain.samples[samples]]
            except IndexError:
                raise ChainSampleError(f"input sample {samples} is not in the chain")
        elif isinstance(samples, range):
            samples = list(samples)
        else:
            raise ChainSampleError("input samples must be an int or a range")

        if len(samples) > chain.nsamples:
            raise ChainSampleError(
                f"samples out of range with chain. chain only has {chain.nsamples} samples"
            )

        if all(isinstance(sample, int) for sample in samples):
            samples = chain._to_chain_sample_format(samples)

        kwargs["samples"] = samples

        return func(*args, **kwargs)

    return wrapper
