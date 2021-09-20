from functools import wraps

import h5py

from cosmoglobe.h5 import PARAMETER_GROUP_NAME
from cosmoglobe.h5.alms import unpack_alms_from_chain
from cosmoglobe.h5.exceptions import ChainKeyError, ChainSampleError


def validate_key(func):
    """Decotrator to check if the requested key exists in the chain."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        chain, key, *_ = args

        root = key.split("/")[0]
        if root.startswith(PARAMETER_GROUP_NAME) or root in chain.samples:
            path = key
        else:
            path = f"{chain.samples[0]}/{key}"

        with h5py.File(chain.path, "r") as file:
            try:
                file[path]
            except KeyError:
                raise ChainKeyError(f"{key=} not found in chain")
        return func(*args, **kwargs)

    return wrapper


def validate_samples(func):
    """Decotrator to check if the samples exist."""

    @wraps(func)
    def wrapper(*args, **kwargs):
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
            samples = chain._format_samples(samples)

        kwargs["samples"] = samples

        return func(*args, **kwargs)

    return wrapper


def unpack_alms(func):
    """Decotrator to that unpacks alms if they key is an alm."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        chain, key, *_ = args
        values = func(*args, **kwargs)

        if "alm" in key:
            return unpack_alms_from_chain(chain, values, key)

        return values

    return wrapper
