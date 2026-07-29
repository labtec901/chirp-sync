"""Chirp detection and decoding from recorded audio.

The pipeline, in order:

1. Resample to a fixed working rate and band-limit to the chirp's band, which
   throws away most of the room's energy before we do anything else.
2. Matched-filter against a single up-chirp and *normalise* by the energy
   actually present in each window.  The result is a correlation coefficient in
   0..1 -- a similarity, not a level -- so one threshold works whether the chirp
   arrived at -6 dBFS from a metre away or at -45 dBFS from across a hall.
3. Average that coefficient at the preamble's symbol spacing.  Eight sweeps
   integrate into one peak, so detection leans on the whole preamble rather
   than trusting any single sweep.
4. Refine the peak back to the *direct path*.  Past a couple of metres the
   loudest correlation lobe is routinely a wall reflection, and since every
   camera sits somewhere different they would each favour a different one --
   whereas the direct sound is the single arrival they all agree about.  A
   reflection can never overtake it, so this is purely a search backwards for
   the earliest lobe that is a real arrival rather than a noise fluctuation,
   which is why the threshold is measured up from the noise floor.
5. Confirm with the down-chirp SFD, which pins the start of the data and cannot
   be faked by the tonal noise -- music, whistles, motor whine -- that can
   otherwise light up a chirp matched filter.
6. Dechirp, FFT, convert bins to LLRs, Viterbi, check CRC.

Detection is CRC-gated: a chirp is only reported if its checksum passes, so a
clip either yields a trustworthy take ID or nothing at all.

Long recordings are analysed in overlapping windows, so peak memory follows the
window size rather than the length of the clip -- camera files run to tens of
minutes and holding several float64 arrays that long would cost gigabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from . import codec, css, frame

FS_WORK = css.FS_WORK
DEFAULT_THRESHOLD = 0.10
# How far back to hunt for the direct path.  Sound covers ~8.5 m in 25 ms, which
# bounds how far ahead of a reflection the direct arrival can plausibly be in a
# room; searching wider mostly invites noise lobes to win.
DIRECT_PATH_LOOKBACK = 0.025
DIRECT_PATH_RATIO = 0.4

# Long files are analysed in overlapping windows so peak memory depends on the
# window, not the recording.  Camera clips run to tens of minutes, and holding
# several float64 arrays the length of an hour of audio would need many
# gigabytes.  The overlap comfortably exceeds the longest burst, so every chirp
# lands wholly inside at least one window.
CHUNK_SECONDS = 90.0
CHUNK_OVERLAP = 25.0


@dataclass
class Detection:
    """One decoded chirp within a piece of audio."""

    time: float                  # seconds from the start of the audio
    take_id: int
    take: str
    profile: str
    score: float                 # normalised preamble correlation, 0..1
    # How far the direct chirp stands above everything else in its band.  Note
    # that "everything else" includes the room's own reverb, not just noise, so
    # a live room reads negative even when the recording sounds fine -- most of
    # the chirp's energy really has arrived smeared rather than direct.
    clarity_db: float
    # Strength of the arrival used for timing relative to the loudest one.
    # Near 1.0 means the direct sound dominated and the sync instant is crisp;
    # a low value means a reverberant field where it is inherently softer.
    direct_ratio: float = 1.0
    repeat_index: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "time": round(self.time, 6),
            "take": self.take,
            "take_id": self.take_id,
            "profile": self.profile,
            "score": round(self.score, 4),
            "clarity_db": round(self.clarity_db, 1),
            "direct_ratio": round(self.direct_ratio, 3),
            "repeat_index": self.repeat_index,
        }


def _resample_to_work(x: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - FS_WORK) < 1e-6:
        return np.asarray(x, dtype=np.float64)
    fs_i = int(round(fs))
    g = int(np.gcd(fs_i, FS_WORK))
    return sps.resample_poly(np.asarray(x, dtype=np.float64), FS_WORK // g,
                             fs_i // g, window=("kaiser", 8.0))


def _bandpass(x: np.ndarray, profile: css.Profile) -> np.ndarray:
    lo = max(20.0, profile.f_low - 200.0)
    hi = min(FS_WORK / 2 - 100.0, profile.f_high + 200.0)
    sos = sps.butter(4, [lo, hi], btype="bandpass", fs=FS_WORK, output="sos")
    return sps.sosfiltfilt(sos, x)


def _matched(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Envelope of a matched filter against a complex chirp template.

    Correlating a real signal against an analytic template already yields the
    complex envelope -- the negative-frequency image lands far out of band --
    so there is no need to Hilbert-transform the whole recording first.

    Overlap-add rather than one big FFT: the template is a couple of thousand
    taps against a signal of millions, so a single transform would allocate
    complex buffers the length of the whole recording for no benefit.  The
    magnitude is taken here so the complex intermediate can be freed at once.
    """
    return np.abs(sps.oaconvolve(x, np.conj(template[::-1]), mode="valid"))


def _window_energy(x: np.ndarray, n: int) -> np.ndarray:
    """Energy of every length-``n`` window, aligned with ``_matched`` output."""
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    return c[n:] - c[:-n]


@dataclass
class _Analysis:
    profile: css.Profile
    filtered: np.ndarray
    rho_up: np.ndarray           # normalised correlation against an up-chirp
    rho_down: np.ndarray         # ... against a down-chirp
    score_up: np.ndarray         # rho_up averaged over the preamble
    score_sfd: np.ndarray        # rho_down averaged over the SFD
    score: np.ndarray            # the two combined


def _analyse(
    work: np.ndarray,
    profile: css.Profile,
    filtered: np.ndarray | None = None,
) -> _Analysis | None:
    nsym = profile.samples_per_symbol(FS_WORK)
    if work.size < nsym * (profile.preamble + profile.sfd + 4):
        return None
    if filtered is None:
        filtered = _bandpass(work, profile)
    up = css.reference_symbol(profile, FS_WORK)
    down = css.reference_symbol(profile, FS_WORK, down=True)

    energy = _window_energy(filtered, nsym)
    # A real chirp of amplitude A gives |corr| = A*nsym/2 and window energy
    # A^2*nsym/2, so this normaliser makes a perfect match come out at exactly 1
    # and anything quieter come out as the fraction of the window it explains.
    scale = np.sqrt(nsym * energy / 2.0)
    floor = 1e-12 + 1e-6 * float(np.sqrt(nsym * max(energy.mean(), 0.0) / 2.0))
    scale = np.maximum(scale, floor)

    rho_up = _matched(filtered, up) / scale
    rho_down = _matched(filtered, down) / scale

    total = profile.preamble + profile.sfd
    usable = min(rho_up.size - (profile.preamble - 1) * nsym,
                 rho_down.size - (total - 1) * nsym)
    if usable <= 0:
        return None

    score_up = np.zeros(usable)
    for k in range(profile.preamble):
        score_up += rho_up[k * nsym: k * nsym + usable]
    score_up /= profile.preamble

    score_sfd = np.zeros(usable)
    for j in range(profile.sfd):
        off = (profile.preamble + j) * nsym
        score_sfd += rho_down[off: off + usable]
    score_sfd /= profile.sfd

    # Scoring the preamble alone leaves a whole-symbol ambiguity: the sweeps are
    # identical, so sliding by one symbol still scores (P-1)/P.  Requiring the
    # down-chirps to line up at the same time breaks the tie, because a
    # one-symbol slip either misses them entirely or catches only half of them.
    score = 0.65 * score_up + 0.35 * score_sfd
    return _Analysis(profile, filtered, rho_up, rho_down, score_up, score_sfd, score)


def _refine_direct_path(mag: np.ndarray, center: int, back: int, fwd: int,
                        floor: float, ratio: float = DIRECT_PATH_RATIO
                        ) -> tuple[float, float]:
    """Locate the direct-path arrival near ``center``; returns ``(position, ratio)``.

    Past a couple of metres the loudest correlation lobe is routinely a wall
    reflection rather than the direct sound.  That is fine for decoding but ruins
    sync between angles, because each camera sits in a different spot and so
    favours a different reflection -- while the direct path is the one arrival
    whose timing every camera agrees about.

    The threshold is measured up from the noise floor, not down from the peak.
    A peak-relative threshold scales with the signal, so in a quiet room it
    correctly ignores noise but in a poor one it starts accepting noise lobes
    that sit at a large fraction of a weak peak.  Anchoring it to the floor
    keeps "is this a real arrival?" a question about SNR, which is what it
    actually is.

    The returned ratio is how strong the chosen arrival is relative to the
    loudest one: near 1.0 the direct path dominates and timing is crisp, while
    a low value means a reverberant field where it is inherently softer.
    """
    lo = max(0, center - back)
    hi = min(mag.size, center + fwd + 1)
    if hi - lo < 3:
        return float(center), 1.0
    window = mag[lo:hi]
    peak_val = float(window.max())
    thresh = floor + ratio * (peak_val - floor)
    inner = window[1:-1]
    local = np.flatnonzero((inner >= window[:-2]) & (inner > window[2:]) & (inner >= thresh))
    i = int(local[0]) + 1 if local.size else int(np.argmax(window))
    strength = float(window[i]) / peak_val if peak_val > 0 else 1.0
    if 0 < i < window.size - 1:
        a, b, c = window[i - 1], window[i], window[i + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return lo + i + delta, strength


def _fractional_shift(x: np.ndarray, delay: float) -> np.ndarray:
    """Delay a signal by a fractional number of samples via an FFT phase ramp."""
    if abs(delay) < 1e-9:
        return x
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    return np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay), n=n)


def _symbol_mags(x: np.ndarray, start: float, profile: css.Profile,
                 count: int, rate_error: float = 0.0) -> np.ndarray:
    """Dechirp ``count`` symbols starting at ``start``.

    ``rate_error`` is the recorder's fractional clock error.  A camera whose
    crystal runs fast records the burst slightly compressed, so the segment is
    read short and stretched back onto the nominal symbol grid before
    demodulation.
    """
    nsym = profile.samples_per_symbol(FS_WORK)
    base = int(np.floor(start))
    if base < 0:
        return np.zeros((0, profile.n))
    frac = start - base
    scale = 1.0 + rate_error
    want = int(np.ceil(count * nsym / scale))
    seg = x[base: base + want + 2 * nsym]
    if seg.size < want:
        return np.zeros((0, profile.n))
    if frac:
        seg = _fractional_shift(seg, -frac)
    if rate_error:
        seg = sps.resample(seg, int(round(seg.size * scale)))
    return css.demodulate(css.to_baseband(seg, profile), profile, count)


def _preamble_mags(x: np.ndarray, t0: float, profile: css.Profile) -> np.ndarray:
    """Dechirp the middle of the preamble, skipping its ragged first sweep."""
    nsym = profile.samples_per_symbol(FS_WORK)
    return _symbol_mags(x, t0 + nsym, profile, max(1, profile.preamble - 2))


def _fine_align(x: np.ndarray, t0: float, profile: css.Profile) -> float:
    """Measure the residual timing error against the preamble, in samples.

    Dechirping a *known* symbol and reading which FFT bin lights up measures the
    timing error directly: sampling d chips late turns preamble symbol 0 into
    symbol d.  This matters more than it sounds -- a chirp symbol is only 2^SF
    chips wide, so being half a chip out (62 microseconds here) already smears
    every symbol's peak across two bins and the whole block fails at once.  The
    correlation peak alone is not precise enough to guarantee that, which is why
    this second stage exists.

    Because every preamble sweep is identical, a window straddling two of them
    still dechirps cleanly, so this works even before the timing is right.
    """
    mags = _preamble_mags(x, t0, profile)
    if mags.shape[0] == 0:
        return t0
    avg = mags.mean(axis=0)
    n = profile.n
    k = int(np.argmax(avg))
    # Parabolic interpolation across the peak, wrapping at the FFT's edges.
    a, b, c = avg[(k - 1) % n], avg[k], avg[(k + 1) % n]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    chips = k + delta
    if chips > n / 2:
        chips -= n
    samples_per_chip = FS_WORK / profile.bw
    return t0 - chips * samples_per_chip


def _block_llrs(mags: np.ndarray, profile: css.Profile,
                chan: np.ndarray | None) -> np.ndarray:
    if chan is None:
        return css.symbol_llrs(mags, profile.sf).ravel()
    return css.symbol_llrs(css.rake_combine(mags, chan), profile.sf,
                           is_power=True).ravel()


def _candidate_channels(chan: np.ndarray | None) -> list[np.ndarray | None]:
    """RAKE first, then plain peak picking; the CRC picks the winner."""
    return [chan, None] if chan is not None else [None]


def _decode_at(x: np.ndarray, start: float, profile: css.Profile,
               chan: np.ndarray | None = None) -> dict | None:
    """Try to decode a frame whose data begins at sample ``start`` (float).

    Both the RAKE combiner and plain peak picking are attempted.  RAKE wins in a
    live room but a clean, near-field recording gives a one-tap channel where
    the plain path is marginally better, and the CRC decides between them for
    free.
    """
    data_syms = frame.block_symbols(codec.PAYLOAD_BYTES, profile.sf)
    mags = _symbol_mags(x, start, profile, data_syms)
    if mags.shape[0] < data_syms:
        return None

    for candidate in _candidate_channels(chan):
        llrs = _block_llrs(mags, profile, candidate)
        data = frame.decode_block(llrs, codec.PAYLOAD_BYTES, profile.sf)
        payload = codec.decode_payload(data) if data is not None else None
        if payload is not None:
            return {"take_id": payload.take_id}
    return None


def _estimate_clarity_db(score: float) -> float:
    """Direct-chirp energy against everything else in the band, in dB.

    The correlation coefficient is the fraction of the window explained by a
    clean chirp, so ``rho^2 / (1 - rho^2)`` is the ratio of that coherent energy
    to all the rest -- background noise and the chirp's own reverberant tail
    alike.  It is a clarity measure rather than an SNR, and a perfectly usable
    recording in a live room will read below 0 dB.
    """
    r2 = min(max(score, 0.0), 0.999999) ** 2
    return float(np.clip(10 * np.log10(r2 / (1.0 - r2)), -40.0, 60.0))


def detect(
    x: np.ndarray,
    fs: float,
    *,
    profiles: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_candidates: int = 64,
) -> list[Detection]:
    """Find and decode every chirp in a mono audio signal.

    Returns one :class:`Detection` per chirp, in time order.  A clip that heard
    several takes -- a camera left rolling between them -- yields several.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)
    if x.size < FS_WORK // 4:
        return []
    work = _resample_to_work(x, fs)

    names = profiles or list(css.PROFILES)
    found: list[Detection] = []

    # All built-in profiles occupy the same band. Filtering is a substantial
    # part of a long scan, so share it rather than doing identical work once per
    # spreading factor. The cache key keeps custom profiles with other bands
    # correct too.
    for base, window in _windows(work):
        filtered_by_band: dict[tuple[float, float], np.ndarray] = {}
        for name in names:
            profile = css.get_profile(name)
            nsym = profile.samples_per_symbol(FS_WORK)
            if window.size < nsym * (profile.preamble + profile.sfd + 4):
                continue
            band = (profile.f_low, profile.f_high)
            filtered = filtered_by_band.get(band)
            if filtered is None:
                filtered = _bandpass(window, profile)
                filtered_by_band[band] = filtered
            _detect_window(
                window, base, profile, threshold, max_candidates, found,
                filtered=filtered)

    found.sort(key=lambda d: d.time)
    seen: dict[int, int] = {}
    for det in found:
        det.repeat_index = seen.get(det.take_id, 0)
        seen[det.take_id] = det.repeat_index + 1
    return found


def _windows(work: np.ndarray):
    """Yield ``(start_sample, view)`` for overlapping analysis windows."""
    size = int(CHUNK_SECONDS * FS_WORK)
    overlap = int(CHUNK_OVERLAP * FS_WORK)
    if work.size <= size:
        yield 0, work
        return
    step = size - overlap
    start = 0
    while start < work.size:
        yield start, work[start:start + size]
        if start + size >= work.size:
            break
        start += step


def _detect_window(work: np.ndarray, base: int, profile: css.Profile,
                   threshold: float, max_candidates: int,
                   found: list[Detection],
                   filtered: np.ndarray | None = None) -> None:
    """Analyse one window and append any chirps it contains to ``found``.

    ``base`` is the window's offset into the whole recording; reported times are
    shifted by it so they stay absolute.
    """
    an = _analyse(work, profile, filtered)
    if an is None:
        return
    nsym = profile.samples_per_symbol(FS_WORK)
    back = max(4, int(DIRECT_PATH_LOOKBACK * FS_WORK))
    fwd = max(2, int(0.002 * FS_WORK))
    guard = nsym * (profile.preamble + profile.sfd)
    # What the score reads when no chirp is present: the reference the
    # direct-path threshold is measured up from.
    noise_floor = float(np.percentile(an.score, 25))

    peaks, _ = sps.find_peaks(an.score, height=threshold, distance=max(1, guard))
    if peaks.size == 0:
        return
    order = np.argsort(an.score[peaks])[::-1][:max_candidates]

    for idx in sorted(int(peaks[i]) for i in order):
        # The SFD must be present, not merely nearby: a lone tonal artefact can
        # light up a chirp matched filter but will not also produce two
        # down-sweeps in exactly the right place.
        if an.score_sfd[idx] < max(0.06, 0.35 * an.score_up[idx]):
            continue

        # Two different anchors, for two different jobs.  Demodulation wants the
        # alignment that best matches the signal actually present, even if that
        # is dominated by a reflection; the *reported* sync instant wants the
        # direct path, because that is the only arrival common to every camera.
        t_demod = _fine_align(an.filtered, float(idx), profile)
        t_sync, direct_ratio = _refine_direct_path(
            an.score, idx, back, fwd, noise_floor)

        # Re-read the preamble at the corrected alignment; that both sharpens
        # the channel estimate and keeps it in the same frame as the data.
        chan = css.channel_profile(_preamble_mags(an.filtered, t_demod, profile))

        data_start = t_demod + (profile.preamble + profile.sfd) * nsym
        chip = FS_WORK / profile.bw
        decoded = None
        for nudge in (0.0, -0.5 * chip, 0.5 * chip, -chip, chip):
            decoded = _decode_at(an.filtered, data_start + nudge, profile, chan)
            if decoded:
                break
        if not decoded:
            continue

        sync_time = (base + t_sync) / FS_WORK
        # Overlapping windows see the same chirp twice, and so can two profiles.
        if any(abs(sync_time - d.time) < 0.05 for d in found):
            continue
        score = float(an.score[idx])
        found.append(
            Detection(
                time=sync_time,
                take_id=decoded["take_id"],
                take=codec.take_id_to_str(decoded["take_id"]),
                profile=profile.name,
                score=score,
                clarity_db=_estimate_clarity_db(score),
                direct_ratio=direct_ratio,
            )
        )
    return found
