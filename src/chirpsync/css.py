"""Chirp spread spectrum modem.

Each symbol is a linear sweep across the whole band, cyclically shifted by the
symbol's value.  That gives three properties this application needs:

* **Processing gain.**  A symbol carries SF bits but occupies 2^SF chips, so the
  demodulator pulls roughly 10*log10(2^SF) dB out of the noise.  At SF=7 that is
  ~21 dB before any coding gain, which is what lets a phone speaker across a
  room beat the HVAC.
* **Immunity to room colouration.**  Every symbol touches every frequency in the
  band for the same amount of time, so a standing-wave null or a cheap mic's
  midrange dip attenuates all symbols equally instead of destroying whichever
  symbol happened to live at that frequency.
* **A sharp sync peak.**  Matched filtering a 4 kHz-wide sweep gives a
  correlation peak ~0.25 ms wide, and interpolating it lands well inside a tenth
  of a millisecond -- three orders of magnitude finer than a video frame.

The band is 1-5 kHz on purpose: it is where phone speakers, camera mics and
lossy codecs are all at their best, and where a sweep reads to the ear as a
bird-like trill rather than a modem screech.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal as sps

FS_WORK = 16000  # internal analysis rate; 3.2x the top of the band


@dataclass(frozen=True)
class Profile:
    """A named set of modem parameters.

    ``sf`` trades duration for robustness: every +1 doubles symbol length and
    buys about 3 dB against noise -- but far more than that against
    reverberation, because a longer symbol means more of the room's echo falls
    inside the symbol that produced it, where the receiver can recover it,
    rather than smearing into the next one, where it is just interference.

    The same ``sf`` is used for the preamble, SFD, and fixed take-ID block.
    """

    name: str
    sf: int = 8
    bw: float = 4000.0
    f_low: float = 1000.0
    preamble: int = 8
    sfd: int = 2

    def with_sf(self, sf: int) -> "Profile":
        return replace(self, sf=sf)

    @property
    def n(self) -> int:
        return 1 << self.sf

    @property
    def symbol_time(self) -> float:
        return self.n / self.bw

    @property
    def f_center(self) -> float:
        return self.f_low + self.bw / 2.0

    @property
    def f_high(self) -> float:
        return self.f_low + self.bw

    def samples_per_symbol(self, fs: float) -> int:
        return int(round(self.symbol_time * fs))


PROFILES = {
    "fast": Profile("fast", sf=7),
    "balanced": Profile("balanced", sf=8),
    "robust": Profile("robust", sf=9),
}
DEFAULT_PROFILE = "fast"


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose from {', '.join(PROFILES)}"
        ) from None


# --- Waveform synthesis ------------------------------------------------------


def _symbol_frequency(profile: Profile, value: int, down: bool, fs: float,
                      sf: int | None = None) -> np.ndarray:
    """Instantaneous frequency of one symbol, in Hz."""
    p = profile if sf is None or sf == profile.sf else profile.with_sf(sf)
    ns = p.samples_per_symbol(fs)
    t = np.arange(ns, dtype=np.float64) / ns  # fraction of the symbol elapsed
    phase_frac = (t + value / p.n) % 1.0
    if down:
        phase_frac = (1.0 - phase_frac) % 1.0
    return p.f_low + p.bw * phase_frac


def _integrate_phase(freq: np.ndarray, fs: float) -> np.ndarray:
    """Trapezoidal phase integration.

    Exact for the piecewise-linear frequency ramps a chirp is made of, and it
    keeps phase continuous across the sweep's wrap point and across symbol
    boundaries -- which is what makes the burst spectrally clean and keeps it
    from clicking.
    """
    step = np.empty_like(freq)
    step[0] = 0.0
    step[1:] = 0.5 * (freq[1:] + freq[:-1])
    return 2.0 * np.pi * np.cumsum(step) / fs


def symbol_sequence_frequency(
    profile: Profile, items: list[tuple[str, int, int]], fs: float
) -> np.ndarray:
    """Instantaneous frequency of a whole burst, on an exact time axis.

    Symbol boundaries are computed in seconds and each output sample asks which
    symbol it falls in, rather than each symbol being rendered as its own
    rounded number of samples.  That distinction matters: a symbol is 2^SF/BW
    seconds, which is a whole number of samples at 48 kHz but *not* at 44.1 kHz,
    so rounding per symbol would let the burst drift against the receiver's grid
    -- about a chip and a half by the end of a frame, which is enough to lose
    the tail of it.
    """
    if not items:
        return np.zeros(0)
    times = np.array([profile.with_sf(sf).symbol_time for _, _, sf in items])
    bounds = np.concatenate([[0.0], np.cumsum(times)])
    n = int(round(bounds[-1] * fs))
    t = np.arange(n, dtype=np.float64) / fs

    idx = np.clip(np.searchsorted(bounds, t, side="right") - 1, 0, len(items) - 1)
    values = np.array([v for _, v, _ in items], dtype=np.float64)
    chips = np.array([1 << sf for _, _, sf in items], dtype=np.float64)
    downs = np.array([kind == "down" for kind, _, _ in items])

    local = (t - bounds[idx]) / times[idx]
    frac = (local + values[idx] / chips[idx]) % 1.0
    frac = np.where(downs[idx], (1.0 - frac) % 1.0, frac)
    return profile.f_low + profile.bw * frac


def synthesize(
    profile: Profile, items: list[tuple[str, int, int]], fs: float, fade: float = 0.004
) -> np.ndarray:
    """Render a symbol list -- ``(kind, value, spreading_factor)`` -- to audio."""
    freq = symbol_sequence_frequency(profile, items, fs)
    wave = np.cos(_integrate_phase(freq, fs))
    nf = int(round(fade * fs))
    if nf > 0 and wave.size > 2 * nf:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        wave[:nf] *= ramp
        wave[-nf:] *= ramp[::-1]
    return wave


def reference_symbol(profile: Profile, fs: float, down: bool = False) -> np.ndarray:
    """Complex reference chirp used for matched filtering and dechirping."""
    freq = _symbol_frequency(profile, 0, down, fs)
    return np.exp(1j * _integrate_phase(freq, fs))


def baseband_reference(profile: Profile) -> np.ndarray:
    """Base up-chirp at complex baseband, sampled at exactly ``bw``."""
    n = profile.n
    t = np.arange(n, dtype=np.float64) / n
    freq = profile.bw * (t - 0.5)
    return np.exp(1j * _integrate_phase(freq, profile.bw))


# --- Symbol mapping ----------------------------------------------------------


def gray_encode_table(sf: int) -> np.ndarray:
    v = np.arange(1 << sf, dtype=np.int64)
    return v ^ (v >> 1)


def bits_to_symbols(bits: np.ndarray, sf: int) -> np.ndarray:
    """Pack bits MSB-first into Gray-coded symbol values."""
    bits = np.asarray(bits, dtype=np.int64)
    if bits.size % sf:
        raise ValueError("bit count must be a multiple of the spreading factor")
    grouped = bits.reshape(-1, sf)
    weights = 1 << np.arange(sf - 1, -1, -1, dtype=np.int64)
    values = grouped @ weights
    return values ^ (values >> 1)


def symbols_to_bits(symbols: np.ndarray, sf: int) -> np.ndarray:
    """Inverse of :func:`bits_to_symbols` (hard decision)."""
    out = []
    for s in np.asarray(symbols, dtype=np.int64):
        v = int(s)
        mask = v
        while mask:
            mask >>= 1
            v ^= mask
        out.extend((v >> np.arange(sf - 1, -1, -1)) & 1)
    return np.array(out, dtype=np.int8)


def channel_profile(preamble_mags: np.ndarray, keep_ratio: float = 0.02) -> np.ndarray:
    """Power delay profile of the room, expressed in dechirped FFT bins.

    Every preamble sweep is symbol 0, so dechirping one shows the channel
    directly: the direct path lands in bin 0 and each reflection appears in the
    bin matching its delay.  Averaging over the preamble and subtracting the
    noise pedestal leaves a usable picture of the room.
    """
    p = np.mean(np.atleast_2d(np.asarray(preamble_mags, dtype=np.float64)) ** 2, axis=0)
    n = p.size
    p = p - np.median(p)
    peak = p.max() if p.size else 0.0
    if peak <= 0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    p[p < keep_ratio * peak] = 0.0
    total = p.sum()
    return p / total if total > 0 else p


def rake_combine(mags: np.ndarray, profile_power: np.ndarray) -> np.ndarray:
    """Match each symbol's dechirped spectrum against the room's delay profile.

    A reflection arriving *d* chips late puts a copy of symbol *k* into bin
    ``k+d``.  Picking the largest bin therefore tracks whichever path happens to
    be loudest, and in a live room that is often a wall rather than the direct
    sound -- which is why plain detection collapses once the direct-to-
    reverberant ratio goes negative.

    Correlating against the measured delay profile instead sums *all* the paths
    into one score, so the room's echoes stop being interference and start being
    diversity.  It also makes demodulation invariant to whole-chip timing error,
    because a shift moves the profile and the symbol together.
    """
    power = np.atleast_2d(np.asarray(mags, dtype=np.float64)) ** 2
    spec = np.fft.fft(power, axis=1) * np.conj(np.fft.fft(profile_power))[None, :]
    return np.maximum(np.real(np.fft.ifft(spec, axis=1)), 0.0)


def symbol_llrs(mags: np.ndarray, sf: int, is_power: bool = False) -> np.ndarray:
    """Per-bit LLRs from FFT bin magnitudes (or from RAKE scores).

    ``mags`` is ``(2^SF,)`` for one symbol or ``(count, 2^SF)`` for many.  We
    convert to a max-log APP: the score of a bit value is the best score of any
    symbol consistent with it.  Normalising by the symbol's mean makes the
    result scale-free, which matters because camera AGC changes the level
    between -- and sometimes during -- takes.
    """
    arr = np.atleast_2d(np.asarray(mags, dtype=np.float64))
    n = 1 << sf
    energy = arr if is_power else arr ** 2
    mean = energy.mean(axis=1, keepdims=True)
    score = np.divide(energy, mean, out=np.zeros_like(energy), where=mean > 0)
    # Undo Gray coding: score_by_value[:, v] is the score of data value v.
    score_by_value = score[:, gray_encode_table(sf)]
    values = np.arange(n)
    out = np.empty((arr.shape[0], sf))
    for j in range(sf):
        zero = ((values >> (sf - 1 - j)) & 1) == 0
        out[:, j] = score_by_value[:, zero].max(axis=1) - score_by_value[:, ~zero].max(axis=1)
    return out.ravel() if np.ndim(mags) == 1 else out


# --- Demodulation ------------------------------------------------------------


def to_baseband(analytic: np.ndarray, profile: Profile, fs: float = FS_WORK) -> np.ndarray:
    """Mix the band down to complex baseband and decimate to exactly ``bw``.

    Sampling at the bandwidth is what makes the two halves of a wrapped chirp
    alias back on top of each other coherently, so a single FFT bin collects the
    symbol's whole energy.
    """
    n = np.arange(analytic.size)
    mixed = analytic * np.exp(-2j * np.pi * profile.f_center * n / fs)
    ratio = fs / profile.bw
    up, down = 1, int(round(ratio))
    if abs(ratio - down) > 1e-9:
        raise ValueError("working rate must be an integer multiple of the bandwidth")
    return sps.resample_poly(mixed, up, down, window=("kaiser", 8.0))


def demodulate(bb: np.ndarray, profile: Profile, count: int) -> np.ndarray:
    """Dechirp ``count`` symbols starting at ``bb[0]``; returns (count, 2^SF) mags."""
    n = profile.n
    ref = np.conj(baseband_reference(profile))
    avail = bb.size // n
    count = min(count, avail)
    if count <= 0:
        return np.zeros((0, n))
    block = bb[: count * n].reshape(count, n) * ref[None, :]
    return np.abs(np.fft.fft(block, axis=1))
