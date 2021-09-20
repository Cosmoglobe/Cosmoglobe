from typing import TYPE_CHECKING

import h5py
import numpy as np
from numba import njit

from cosmoglobe.h5 import PARAMETER_GROUP_NAME
from cosmoglobe.h5.exceptions import ChainKeyError

if TYPE_CHECKING:
    from cosmoglobe.h5.chain import Chain


def unpack_alms_from_chain(chain: "Chain", alms: np.ndarray, key: str) -> np.ndarray:
    """Unpacks alms from a chain."""

    root = key.split("/")[0]
    if root.startswith(PARAMETER_GROUP_NAME) or root in chain.samples:
        lmax_key = f"{key[:-3]}lmax"
    else:
        lmax_key = f"{chain.samples[0]}/{key[:-3]}lmax"

    try:
        with h5py.File(chain.path, "r") as file:
            lmax = file[lmax_key][()]
    except KeyError:
        raise ChainKeyError(
            f"{lmax_key} does not exist. Cannot unpack alms from chain "
            "without an lmax present in the chain"
        )

    if alms.ndim > 2:  # alms have shape (1, n) or (3, n)
        return np.asarray([unpack_alms(value, lmax) for value in alms])

    return unpack_alms(alms, lmax)


@njit
def unpack_alms(data, lmax):
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
