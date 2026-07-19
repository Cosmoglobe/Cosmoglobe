# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-optimized Huffman decoder for fast bitstring decoding."""
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t

def prepare_tree_decoder_from_tree(np.ndarray tree, np.ndarray symb):
    """Prepare flat Commander Huffman tree arrays for direct decoding."""
    cdef:
        np.ndarray[np.int64_t, ndim=1] tree64 = np.asarray(tree, dtype=np.int64)
        np.ndarray[np.int64_t, ndim=1] symb64 = np.asarray(symb, dtype=np.int64)
        int node_max = <int>tree64[0]
        int nsymb = <int>symb64.shape[0]
        int ninternal = node_max - nsymb

    if node_max <= 0 or nsymb <= 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            0,
            0,
        )

    return (
        np.asarray(tree64[1:1 + ninternal], dtype=np.int32),
        np.asarray(tree64[1 + ninternal:1 + 2 * ninternal], dtype=np.int32),
        symb64,
        node_max,
        nsymb,
    )


def huffman_decode_tree(bytes bytarr,
                        np.ndarray[np.int32_t, ndim=1] left_nodes,
                        np.ndarray[np.int32_t, ndim=1] right_nodes,
                        np.ndarray[np.int64_t, ndim=1] symb,
                        int node_max,
                        int nsymb):
    """Decode Huffman bytes by walking Commander flat tree arrays directly."""
    cdef:
        np.ndarray[np.uint8_t, ndim=1] arr = np.frombuffer(bytarr, dtype=np.uint8)
        np.ndarray[np.int64_t, ndim=1] decoded_arr = np.empty(arr.shape[0] * 8, dtype=np.int64)
        int decode_idx = 0
        int padding
        int byte_idx, nbits, shift
        uint8_t curr_byte
        int last_byte_idx
        int node = node_max
        int arr_len = arr.shape[0]
        int idx

    if arr_len == 0:
        return np.empty(0, dtype=np.int64)

    padding = arr[0]
    last_byte_idx = arr_len - 1

    for byte_idx in range(1, arr_len):
        curr_byte = arr[byte_idx]
        nbits = 8
        if byte_idx == last_byte_idx:
            nbits -= padding

        for shift in range(7, 7 - nbits, -1):
            idx = node - nsymb - 1
            if (curr_byte >> shift) & 1:
                node = right_nodes[idx]
            else:
                node = left_nodes[idx]

            if node <= nsymb:
                decoded_arr[decode_idx] = symb[node - 1]
                decode_idx += 1
                node = node_max

    return decoded_arr[:decode_idx]

