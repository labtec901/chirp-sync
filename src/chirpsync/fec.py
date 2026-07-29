"""Forward error correction, interleaving and whitening.

The encoder side of everything here is mirrored in ``webapp/chirpsync.js`` and
must stay bit-exact with it; ``tests/test_js_parity.py`` checks that.

Chain (transmit order)::

    payload bytes -> whiten -> unpack to bits -> punctured convolutional code
                  -> pad to multiple of SF -> interleave -> symbol mapping

The mother code uses the K=7 generators (0o171, 0o133, 0o165), zero terminated
with 6 tail bits. The third output is omitted every third step, giving an
effective rate of 3/8 with less airtime than the full rate-1/3 code. Decoding is
soft-decision Viterbi driven by log likelihood ratios from the demodulator.
"""

from __future__ import annotations

import numpy as np

K = 7
NSTATES = 1 << (K - 1)
POLYS = (0o171, 0o133, 0o165)
RATE_DEN = len(POLYS)
TAIL = K - 1


def _puncture_mask(nsteps: int) -> np.ndarray:
    """Keep both base outputs and two out of every three extra parity bits."""
    mask = np.ones((nsteps, RATE_DEN), dtype=bool)
    mask[2::3, 2] = False
    return mask.ravel()


def coded_length(n_info: int) -> int:
    """Number of transmitted coded bits for ``n_info`` payload bits."""
    return int(_puncture_mask(n_info + TAIL).sum())


def _depuncture(soft: np.ndarray, n_info: int) -> np.ndarray:
    """Restore omitted parity positions as zero-confidence erasures."""
    mask = _puncture_mask(n_info + TAIL)
    want = int(mask.sum())
    soft = np.asarray(soft, dtype=np.float64)
    if soft.size < want:
        raise ValueError("not enough coded bits for the requested payload")
    out = np.zeros(mask.size, dtype=np.float64)
    out[mask] = soft[:want]
    return out


def _parity(x: int) -> int:
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def _build_trellis():
    """Precompute predecessor states and branch outputs.

    ``ns = (b << (K-2)) | (s >> 1)`` so each next-state has exactly two
    predecessors that differ only in their least significant bit.
    """
    pred = np.zeros((NSTATES, 2), dtype=np.int64)
    outbits = np.zeros((NSTATES, 2, RATE_DEN), dtype=np.int8)
    inbit = np.zeros(NSTATES, dtype=np.int8)
    for ns in range(NSTATES):
        b = ns >> (K - 2)
        inbit[ns] = b
        base = (ns << 1) & (NSTATES - 1)
        for j in (0, 1):
            s = base | j
            pred[ns, j] = s
            reg = (b << (K - 1)) | s
            for p, poly in enumerate(POLYS):
                outbits[ns, j, p] = _parity(reg & poly)
    # Pack each output tuple into a pattern index for fast metric lookup.
    weights = 1 << np.arange(RATE_DEN - 1, -1, -1, dtype=np.int64)
    pattern = outbits.astype(np.int64) @ weights
    return pred, pattern, inbit


_PRED, _PATTERN, _INBIT = _build_trellis()
_PATTERN_SIGNS = 1.0 - 2.0 * (
    (np.arange(1 << RATE_DEN)[:, None]
     >> np.arange(RATE_DEN - 1, -1, -1)) & 1
)


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Punctured K=7 convolutional encode with 6 zero tail bits."""
    bits = np.asarray(bits, dtype=np.int8)
    padded = np.concatenate([bits, np.zeros(TAIL, dtype=np.int8)])
    out = np.empty(padded.size * RATE_DEN, dtype=np.int8)
    state = 0
    for i, b in enumerate(padded):
        reg = (int(b) << (K - 1)) | state
        for p, poly in enumerate(POLYS):
            out[i * RATE_DEN + p] = _parity(reg & poly)
        state = reg >> 1
    return out[_puncture_mask(padded.size)]


def viterbi_decode(soft: np.ndarray, n_info: int) -> np.ndarray:
    """Soft-decision Viterbi.

    ``soft[i]`` is a log likelihood ratio for coded bit *i* using the
    convention ``log P(bit=0) - log P(bit=1)``; magnitude carries confidence,
    so an erased/padded bit is simply 0.  Returns ``n_info`` decoded bits.
    """
    nsteps = n_info + TAIL
    s = _depuncture(soft, n_info).reshape(nsteps, RATE_DEN)

    metric = np.full(NSTATES, -1e30)
    metric[0] = 0.0
    choices = np.zeros((nsteps, NSTATES), dtype=np.uint8)

    # lut[p] is the branch metric for output pattern p, recomputed per step.
    for t in range(nsteps):
        lut = _PATTERN_SIGNS @ s[t]
        cand = metric[_PRED] + lut[_PATTERN]  # (NSTATES, 2)
        pick = np.argmax(cand, axis=1)
        metric = cand[np.arange(NSTATES), pick]
        choices[t] = pick
        # Renormalise to keep the metrics from drifting off into the weeds.
        metric -= metric.max()

    # Zero terminated, so the survivor path ends in state 0.
    state = 0
    bits = np.zeros(nsteps, dtype=np.int8)
    for t in range(nsteps - 1, -1, -1):
        bits[t] = _INBIT[state]
        state = int(_PRED[state, choices[t, state]])
    return bits[:n_info]


# --- Interleaving ------------------------------------------------------------
#
# A stride permutation over the whole coded block. Consecutive parity bits and
# neighbouring trellis steps get scattered across different chirp symbols, so
# losing one symbol to a reflection or a noise burst costs the Viterbi decoder
# isolated errors rather than a contiguous run.


def _stride_for(n: int) -> int:
    if n < 3:
        return 1
    # Start near n/phi and walk up until coprime with n, which makes the
    # permutation a single cycle and therefore invertible.
    stride = max(1, int(n / 1.6180339887))
    while np.gcd(stride, n) != 1:
        stride += 1
        if stride >= n:
            return 1
    return stride


def interleave_indices(n: int) -> np.ndarray:
    stride = _stride_for(n)
    return (np.arange(n, dtype=np.int64) * stride) % n


def interleave(bits: np.ndarray) -> np.ndarray:
    idx = interleave_indices(bits.size)
    return np.asarray(bits)[idx]


def deinterleave(values: np.ndarray) -> np.ndarray:
    idx = interleave_indices(values.size)
    out = np.empty_like(values)
    out[idx] = values
    return out


# --- Whitening ---------------------------------------------------------------
#
# PN9 LFSR (x^9 + x^5 + 1), the same one used by CC1101 and friends.  Keeps long
# runs of identical bytes from turning into long runs of identical chirp
# symbols, which would otherwise confuse the frame detector.


def pn9(nbytes: int) -> np.ndarray:
    state = 0x1FF
    out = np.empty(nbytes, dtype=np.uint8)
    for i in range(nbytes):
        byte = 0
        for b in range(8):
            byte |= (state & 1) << b
            fb = ((state & 1) ^ ((state >> 5) & 1)) & 1
            state = (state >> 1) | (fb << 8)
        out[i] = byte
    return out


def whiten(data: bytes) -> bytes:
    mask = pn9(len(data))
    return bytes(np.frombuffer(data, dtype=np.uint8) ^ mask)
