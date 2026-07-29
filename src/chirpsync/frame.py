"""Frame layout shared by the generator and the parser.

On-air order::

    [ preamble: P up-chirps ] [ SFD: 2 down-chirps ] [ take-ID block ]

The preamble is a run of identical sweeps, which gives the detector a periodic
signature to integrate over; the down-chirps that follow are unmistakable
against it and mark exactly where the fixed-length take-ID block starts.

The sync reference -- the instant every clip of a take gets aligned to -- is
defined as the start of the first preamble sweep.
"""

from __future__ import annotations

import numpy as np

from . import codec, css, fec


def coded_length(n_info: int) -> int:
    return fec.coded_length(n_info)


def block_symbols(n_bytes: int, sf: int) -> int:
    """How many chirp symbols a block of ``n_bytes`` occupies."""
    return -(-coded_length(n_bytes * 8) // sf)


def encode_block(data: bytes, sf: int) -> np.ndarray:
    """Bytes -> chirp symbol values."""
    bits = np.unpackbits(np.frombuffer(fec.whiten(data), dtype=np.uint8))
    coded = fec.conv_encode(bits)
    nsym = -(-coded.size // sf)
    padded = np.zeros(nsym * sf, dtype=np.int8)
    padded[: coded.size] = coded
    return css.bits_to_symbols(fec.interleave(padded), sf)


def decode_block(llrs: np.ndarray, n_bytes: int, sf: int) -> bytes | None:
    """Per-bit LLRs -> bytes, or None if the block is short or corrupt.

    ``llrs`` must be the concatenated per-symbol LLRs for exactly
    ``block_symbols(n_bytes, sf)`` symbols.
    """
    want_sym = block_symbols(n_bytes, sf)
    if llrs.size < want_sym * sf:
        return None
    padded = fec.deinterleave(llrs[: want_sym * sf])
    n_info = n_bytes * 8
    bits = fec.viterbi_decode(padded[: coded_length(n_info)], n_info)
    return fec.whiten(bytes(np.packbits(bits)))


def frame_items(payload: codec.Payload, profile: css.Profile) -> tuple[list, dict]:
    """Build the full symbol list for a payload, plus a layout description.

    Items are ``(kind, value, spreading_factor)``. The entire frame uses the
    profile's single spreading factor.
    """
    data = codec.encode_payload(payload)
    sf = profile.sf
    sym_time = profile.symbol_time

    items: list[tuple[str, int, int]] = [("up", 0, sf)] * profile.preamble
    items += [("down", 0, sf)] * profile.sfd
    for value in encode_block(data, sf):
        items.append(("up", int(value), sf))

    data_symbols = block_symbols(len(data), sf)
    layout = {
        "preamble_symbols": profile.preamble,
        "sfd_symbols": profile.sfd,
        "data_symbols": data_symbols,
        "total_symbols": len(items),
        "duration": (profile.preamble + profile.sfd + data_symbols) * sym_time,
    }
    return items, layout
