"""An acoustic simulator for beating the chirp up in realistic ways.

Nothing here is part of the shipped library; it exists so the modem can be
measured rather than guessed at.  The chain models what actually happens between
a phone screen and a camera's memory card::

    chirp -> speaker response + speaker distortion -> room impulse response
          -> mic response -> preamp hiss -> background noise -> AGC -> clipping
          -> quantisation -> sample clock offset -> lossy codec

Every stage is something that has broken an audio watermarking scheme in the
field at some point, which is why they are all here.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

SPEED_OF_SOUND = 343.0


# --- Room acoustics ----------------------------------------------------------


def _add_impulses(length: int, positions: np.ndarray, amplitudes: np.ndarray,
                  width: int = 8) -> np.ndarray:
    """Sum band-limited impulses at fractional sample positions.

    Rounding reflections to whole samples would quantise their arrival times and
    flatter the detector; a windowed sinc puts each one where it really lands.
    """
    ir = np.zeros(length)
    if positions.size == 0:
        return ir
    centers = np.floor(positions).astype(np.int64)
    frac = positions - centers
    offsets = np.arange(-width, width + 1)
    idx = centers[:, None] + offsets[None, :]
    t = offsets[None, :] - frac[:, None]
    kernel = np.sinc(t) * (0.54 + 0.46 * np.cos(np.pi * np.clip(t / (width + 1), -1, 1)))
    vals = amplitudes[:, None] * kernel
    keep = (idx >= 0) & (idx < length)
    return np.bincount(idx[keep], weights=vals[keep], minlength=length)[:length]


_RIR_CACHE: dict = {}


def room_ir(
    fs: float,
    *,
    room: tuple[float, float, float] = (6.0, 5.0, 3.0),
    src: tuple[float, float, float] = (1.5, 1.5, 1.5),
    mic: tuple[float, float, float] = (4.0, 3.0, 1.4),
    rt60: float = 0.45,
    order: int = 8,
    diffuse: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Shoebox room impulse response; returns ``(ir, direct_path_samples)``.

    Image sources (Allen & Berkley) give physically correct early reflections --
    the part that actually confuses a matched filter -- and a decaying noise tail
    stands in for the diffuse late field that a finite reflection order misses.
    """
    key = (round(fs), tuple(np.round(room, 3)), tuple(np.round(src, 3)),
           tuple(np.round(mic, 3)), round(rt60, 4), order, diffuse, seed)
    hit = _RIR_CACHE.get(key)
    if hit is not None:
        return hit

    rng = np.random.default_rng(seed)
    room_a = np.asarray(room, float)
    src_a = np.asarray(src, float)
    mic_a = np.asarray(mic, float)

    volume = float(np.prod(room_a))
    surface = 2.0 * (room_a[0] * room_a[1] + room_a[1] * room_a[2] + room_a[0] * room_a[2])
    # Sabine, inverted for the absorption coefficient implied by the target RT60.
    alpha = float(np.clip(24.0 * np.log(10.0) * volume / (SPEED_OF_SOUND * rt60 * surface),
                          0.01, 0.95))
    beta = np.sqrt(1.0 - alpha)

    direct = float(np.linalg.norm(src_a - mic_a))
    direct_samples = direct / SPEED_OF_SOUND * fs
    length = int((rt60 * 1.5 + direct / SPEED_OF_SOUND) * fs) + 64

    n = np.arange(-order, order + 1)
    nx, ny, nz, qx, qy, qz = np.meshgrid(n, n, n, [0, 1], [0, 1], [0, 1], indexing="ij")
    nx, ny, nz = nx.ravel(), ny.ravel(), nz.ravel()
    qx, qy, qz = qx.ravel(), qy.ravel(), qz.ravel()

    px = (1 - 2 * qx) * src_a[0] + 2 * nx * room_a[0]
    py = (1 - 2 * qy) * src_a[1] + 2 * ny * room_a[1]
    pz = (1 - 2 * qz) * src_a[2] + 2 * nz * room_a[2]
    refl = (np.abs(nx - qx) + np.abs(nx) + np.abs(ny - qy) + np.abs(ny)
            + np.abs(nz - qz) + np.abs(nz))

    dist = np.sqrt((px - mic_a[0]) ** 2 + (py - mic_a[1]) ** 2 + (pz - mic_a[2]) ** 2)
    delay = dist / SPEED_OF_SOUND * fs
    keep = (refl <= order) & (dist > 1e-6) & (delay < length - 16)
    ir = _add_impulses(length, delay[keep], (beta ** refl[keep]) / dist[keep])

    if diffuse:
        # Blend in an exponentially decaying noise tail past the mixing time so
        # the late field has the density a real room does.
        t = np.arange(length) / fs
        t_mix = min(0.03, rt60 / 4)
        decay = 10 ** (-3.0 * t / rt60)
        tail = rng.standard_normal(length) * decay
        tail[t < t_mix] = 0.0
        i0 = int(t_mix * fs)
        i1 = i0 + max(1, int(0.01 * fs))
        early_ref = np.sqrt(np.mean(ir[i0:i1] ** 2)) + 1e-15
        tail_ref = np.sqrt(np.mean(tail[i0:i1] ** 2)) + 1e-15
        ir = ir + tail * (early_ref / tail_ref)

    # Frequency-dependent decay: rooms absorb treble faster than bass, so the
    # high band's reverb tail is shorter than the broadband RT60.
    ir = _band_decay(ir, fs, rt60)
    ir /= np.max(np.abs(ir)) + 1e-12
    result = (ir, direct_samples)
    if len(_RIR_CACHE) < 256:
        _RIR_CACHE[key] = result
    return result


def _band_decay(ir: np.ndarray, fs: float, rt60: float) -> np.ndarray:
    """Shorten the reverb tail at high frequencies, as real surfaces do."""
    t = np.arange(ir.size) / fs
    bands = [
        (None, 500.0, rt60 * 1.20),
        (500.0, 2000.0, rt60 * 1.00),
        (2000.0, 6000.0, rt60 * 0.75),
        (6000.0, None, rt60 * 0.50),
    ]
    out = np.zeros_like(ir)
    nyq = fs / 2
    for lo, hi, band_rt in bands:
        if lo is None:
            sos = sps.butter(4, hi / nyq, btype="low", output="sos")
        elif hi is None or hi >= nyq * 0.99:
            sos = sps.butter(4, lo / nyq, btype="high", output="sos")
        else:
            sos = sps.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        part = sps.sosfilt(sos, ir)
        # Extra decay relative to what the image-source model already applied.
        out += part * 10 ** (-3.0 * t * (1.0 / band_rt - 1.0 / rt60))
    return out


def apply_room(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    return sps.fftconvolve(x, ir, mode="full")[: x.size + ir.size - 1]


# --- Transducers -------------------------------------------------------------


def _peaking(fs: float, f0: float, gain_db: float, q: float) -> np.ndarray:
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    ai = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return sps.tf2sos(np.array(b) / ai[0], np.array(ai) / ai[0])


def _shelf_or_pass(fs: float, f0: float, kind: str, order: int = 2) -> np.ndarray:
    return sps.butter(order, f0 / (fs / 2), btype=kind, output="sos")


MIC_MODELS = {
    "flat": [],
    "camera_internal": [
        ("hp", 160.0, 2), ("peak", 1000.0, -4.0, 1.2), ("peak", 4500.0, 4.0, 1.5),
        ("lp", 9000.0, 2),
    ],
    "phone_bottom": [
        ("hp", 320.0, 2), ("peak", 3000.0, 6.0, 2.5), ("peak", 800.0, -5.0, 1.0),
        ("lp", 7500.0, 3),
    ],
    "lavalier": [("hp", 90.0, 2), ("peak", 6000.0, 5.0, 1.0), ("lp", 12000.0, 2)],
    "shotgun": [("hp", 80.0, 2), ("peak", 5000.0, 3.0, 0.9)],
    "cheap_usb": [
        ("hp", 120.0, 2), ("peak", 220.0, 5.0, 1.0), ("peak", 2500.0, -6.0, 1.4),
        ("lp", 6000.0, 3),
    ],
    "action_cam": [
        ("hp", 300.0, 3), ("peak", 1800.0, -7.0, 1.0), ("peak", 3500.0, 5.0, 2.0),
        ("lp", 8000.0, 3),
    ],
}

SPEAKER_MODELS = {
    "studio": [],
    "phone": [("hp", 600.0, 4), ("peak", 1300.0, 6.0, 1.5), ("peak", 3000.0, -4.0, 1.5),
              ("lp", 11000.0, 2)],
    "laptop": [("hp", 350.0, 3), ("peak", 800.0, -5.0, 1.2), ("peak", 5000.0, 3.0, 1.0),
               ("lp", 13000.0, 2)],
    "bt_speaker": [("hp", 150.0, 2), ("peak", 2000.0, -3.0, 1.0), ("lp", 14000.0, 2)],
    "tablet": [("hp", 450.0, 3), ("peak", 1600.0, 4.0, 1.8), ("lp", 12000.0, 2)],
}


def _build_chain(fs: float, spec: list) -> list[np.ndarray]:
    out = []
    for item in spec:
        if item[0] == "peak":
            out.append(_peaking(fs, item[1], item[2], item[3]))
        elif item[0] == "hp":
            out.append(_shelf_or_pass(fs, item[1], "high", item[2]))
        elif item[0] == "lp":
            out.append(_shelf_or_pass(fs, min(item[1], fs / 2 * 0.98), "low", item[2]))
    return out


def colour(x: np.ndarray, fs: float, spec: list) -> np.ndarray:
    for sos in _build_chain(fs, spec):
        x = sps.sosfilt(sos, x)
    return x


def speaker_distortion(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Soft saturation plus a little second harmonic, as small drivers do."""
    if drive <= 0:
        return x
    y = np.tanh(x * (1.0 + 3.0 * drive)) / (1.0 + 3.0 * drive)
    y = y + 0.06 * drive * (y ** 2 - np.mean(y ** 2))
    return y * (np.max(np.abs(x)) / (np.max(np.abs(y)) + 1e-12))


# --- Noise beds --------------------------------------------------------------


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if f.size > 1 else 1.0
    return np.fft.irfft(spec / np.sqrt(f), n=n)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if f.size > 1 else 1.0
    return np.fft.irfft(spec / f, n=n)


def _babble(n: int, fs: float, rng: np.random.Generator, voices: int = 8) -> np.ndarray:
    """Overlapping voices: the hardest realistic masker for a 1-5 kHz signal."""
    out = np.zeros(n)
    t = np.arange(n) / fs
    for _ in range(voices):
        base = _pink(n, rng)
        # Two formants plus a syllabic envelope at a speech-like rate.
        f1 = rng.uniform(400, 900)
        f2 = rng.uniform(1200, 2800)
        v = sps.sosfilt(_peaking(fs, f1, 12.0, 2.0), base)
        v = sps.sosfilt(_peaking(fs, f2, 10.0, 2.5), v)
        v = sps.sosfilt(sps.butter(2, [150 / (fs / 2), 6000 / (fs / 2)],
                                   btype="band", output="sos"), v)
        rate = rng.uniform(2.5, 5.0)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28))
        env = env ** 2
        out += v * env
    return out


def _hvac(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    base = sps.sosfilt(sps.butter(2, 900 / (fs / 2), btype="low", output="sos"),
                       _pink(n, rng))
    t = np.arange(n) / fs
    hum = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / (i + 1)
              for i, f in enumerate((50.0, 100.0, 150.0, 200.0)))
    return base + 0.25 * hum


def _street(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    # Mostly low-frequency rumble, but real traffic rolls off nearer 6 dB/oct
    # than 12; pure brown noise makes an unrealistically bottom-heavy bed.
    out = _brown(n, rng) * 1.2 + _pink(n, rng) * 0.8
    # A handful of vehicle passes: broadband, slow envelope, plenty of midrange.
    for _ in range(max(1, n // int(fs * 4))):
        start = rng.integers(0, max(1, n - int(fs * 2)))
        dur = int(fs * rng.uniform(0.8, 2.0))
        seg = min(dur, n - start)
        env = np.hanning(dur)[:seg]
        burst = sps.sosfilt(
            sps.butter(2, [200 / (fs / 2), 4000 / (fs / 2)], btype="band", output="sos"),
            rng.standard_normal(seg))
        out[start:start + seg] += burst * env * 1.5
    return out


def _music(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Chords with harmonics plus hats -- lots of energy right in the chirp band."""
    t = np.arange(n) / fs
    out = np.zeros(n)
    roots = [110.0, 146.83, 164.81, 130.81]
    bar = int(fs * 1.2)
    for i, start in enumerate(range(0, n, bar)):
        seg = min(bar, n - start)
        tt = t[:seg]
        root = roots[i % len(roots)]
        env = np.exp(-2.5 * tt) + 0.15
        chord = sum(np.sin(2 * np.pi * root * r * h * tt + rng.uniform(0, 6.28)) / h
                    for r in (1.0, 1.26, 1.5) for h in (1, 2, 3, 4, 5, 6))
        out[start:start + seg] += chord * env
    # Hi-hats every eighth note: bright, transient, right on top of the band.
    step = int(fs * 0.3)
    for start in range(0, n, step):
        seg = min(int(fs * 0.08), n - start)
        if seg <= 0:
            continue
        hat = sps.sosfilt(sps.butter(2, 4000 / (fs / 2), btype="high", output="sos"),
                          rng.standard_normal(seg))
        out[start:start + seg] += hat * np.exp(-40 * np.arange(seg) / fs) * 4.0
    return out


NOISE_KINDS = ("white", "pink", "babble", "hvac", "street", "music")


def noise_bed(kind: str, n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        out = rng.standard_normal(n)
    elif kind == "pink":
        out = _pink(n, rng)
    elif kind == "babble":
        out = _babble(n, fs, rng)
    elif kind == "hvac":
        out = _hvac(n, fs, rng)
    elif kind == "street":
        out = _street(n, fs, rng)
    elif kind == "music":
        out = _music(n, fs, rng)
    else:
        raise ValueError(f"unknown noise kind {kind!r}")
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)


# --- Level, SNR and dynamics -------------------------------------------------


def band_rms(x: np.ndarray, fs: float, lo: float = 1000.0, hi: float = 5000.0) -> float:
    """RMS inside the chirp's band -- the only SNR that means anything here."""
    hi = min(hi, fs / 2 * 0.98)
    sos = sps.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    y = sps.sosfilt(sos, x)
    return float(np.sqrt(np.mean(y ** 2)))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, fs: float, snr_db: float,
               window: tuple[int, int] | None = None) -> np.ndarray:
    """Add ``noise`` to ``signal`` at a given in-band SNR.

    The signal's level is measured only over the chirp itself; measuring across
    the silence around it would understate the level and quietly make every test
    easier than advertised.
    """
    lo, hi = window if window else (0, signal.size)
    s = band_rms(signal[lo:hi], fs)
    nz = band_rms(noise, fs)
    if s <= 0 or nz <= 0:
        return signal + noise
    target = s / (10 ** (snr_db / 20.0))
    return signal + noise * (target / nz)


def agc(x: np.ndarray, fs: float, target: float = 0.15, attack: float = 0.005,
        release: float = 0.4, max_gain_db: float = 30.0) -> np.ndarray:
    """A camera's automatic gain control, complete with pumping.

    Fast attack and slow release means a chirp arriving after quiet gets its
    leading edge squashed -- exactly the part the sync estimate depends on.
    """
    # Real AGCs update per block, not per sample; running the smoother at a 1 kHz
    # control rate matches that and keeps this out of the profiler.
    hop = max(1, int(fs / 1000))
    env = np.abs(sps.sosfilt(sps.butter(2, 30 / (fs / 2), btype="low", output="sos"),
                             np.abs(x)))[::hop]
    a_att = np.exp(-hop / (attack * fs))
    a_rel = np.exp(-hop / (release * fs))
    max_gain = 10 ** (max_gain_db / 20.0)
    want = np.minimum(target / np.maximum(env, 1e-6), max_gain)
    coarse = np.empty_like(want)
    g = 1.0
    for i, w in enumerate(want):
        coeff = a_att if w < g else a_rel
        g = coeff * g + (1 - coeff) * w
        coarse[i] = g
    gain = np.interp(np.arange(x.size), np.arange(coarse.size) * hop, coarse)
    return x * gain


def clip(x: np.ndarray, ceiling: float = 1.0, soft: bool = True) -> np.ndarray:
    if soft:
        return ceiling * np.tanh(x / ceiling)
    return np.clip(x, -ceiling, ceiling)


def quantise(x: np.ndarray, bits: int = 16) -> np.ndarray:
    step = 2.0 ** -(bits - 1)
    return np.round(x / step) * step


def clock_offset(x: np.ndarray, ppm: float) -> np.ndarray:
    """Resample as if the recorder's crystal ran ``ppm`` parts-per-million fast."""
    if abs(ppm) < 1e-9:
        return x
    n_out = int(round(x.size * (1.0 + ppm * 1e-6)))
    return sps.resample(x, n_out)


def dropout(x: np.ndarray, fs: float, starts: list[float], length: float) -> np.ndarray:
    y = x.copy()
    n = int(length * fs)
    for s in starts:
        i = int(s * fs)
        y[i:i + n] = 0.0
    return y


# --- Lossy codecs ------------------------------------------------------------


def codec_roundtrip(x: np.ndarray, fs: float, codec: str = "aac",
                    bitrate: str = "96k") -> np.ndarray:
    """Encode and decode through ffmpeg, returning the same number of samples.

    Codec priming samples are trimmed back off so the result stays aligned with
    the input; if a codec ever fails to round-trip alignment, the sync tests
    below will show it as a constant bias rather than hiding it.
    """
    import soundfile as sf

    ext = {"aac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg"}[codec]
    enc = {"aac": "aac", "mp3": "libmp3lame", "opus": "libopus", "vorbis": "libvorbis"}[codec]
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        mid = os.path.join(td, f"mid.{ext}")
        dst = os.path.join(td, "out.wav")
        sf.write(src, x, int(fs), subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-c:a", enc, "-b:a", bitrate, mid], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mid,
                        "-ar", str(int(fs)), "-ac", "1", dst], check=True)
        y, _ = sf.read(dst)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size)])
    return y[: x.size]


# --- The whole chain ---------------------------------------------------------


@dataclass
class Condition:
    """One realistic recording situation."""

    name: str
    rt60: float = 0.35
    distance: float = 2.0
    room: tuple[float, float, float] = (6.0, 5.0, 3.0)
    noise: str = "hvac"
    snr_db: float = 20.0
    mic: str = "camera_internal"
    speaker: str = "phone"
    drive: float = 0.3
    use_agc: bool = False
    clip_ceiling: float | None = None
    codec: tuple[str, str] | None = None
    ppm: float = 0.0
    bits: int = 16
    # Where the mix is normalised to before any AGC or clipping.  Whoever set
    # the levels left themselves headroom; without this, a bass-heavy noise bed
    # scaled to hit an in-band SNR target can drive the whole file into
    # saturation and the test stops measuring the thing it claims to.
    peak_dbfs: float = -6.0
    preamp_db: float = -70.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class Capture:
    audio: np.ndarray
    fs: float
    true_time: float          # when the sync reference really arrived, in seconds
    condition: Condition


def simulate(chirp: np.ndarray, fs: float, sync_offset: float,
             cond: Condition, *, burst_duration: float | None = None,
             seed: int | None = None) -> Capture:
    """Run a generated chirp through one recording situation."""
    rng = np.random.default_rng(cond.seed if seed is None else seed)

    # Place the source and mic to realise the requested distance in the room.
    room = np.asarray(cond.room, float)
    src = np.array([room[0] * 0.25, room[1] * 0.4, 1.4])
    direction = np.array([1.0, 0.55, 0.0])
    direction /= np.linalg.norm(direction)
    mic = src + direction * cond.distance
    mic = np.minimum(np.maximum(mic, 0.3), room - 0.3)
    mic[2] = min(1.5, room[2] - 0.4)

    x = np.asarray(chirp, float)
    x = colour(x, fs, SPEAKER_MODELS[cond.speaker])
    x = speaker_distortion(x, cond.drive)

    ir, direct_samples = room_ir(fs, room=tuple(room), src=tuple(src), mic=tuple(mic),
                                 rt60=cond.rt60, seed=int(rng.integers(1 << 30)))
    wet = apply_room(x, ir)

    # Level: a phone at 1 m lands around -20 dBFS on a camera; inverse square
    # from there.
    ref = 10 ** (-20 / 20.0)
    wet = wet / (np.max(np.abs(wet)) + 1e-12) * ref * min(1.0, 1.5 / max(cond.distance, 0.3))

    wet = colour(wet, fs, MIC_MODELS[cond.mic])

    # Measure the signal over the burst alone.  Including the silence padded
    # around it would understate the level and quietly make every test easier
    # than its label claims.
    burst = burst_duration if burst_duration else (len(chirp) / fs - sync_offset)
    chirp_lo = int(sync_offset * fs + direct_samples)
    chirp_hi = min(wet.size, chirp_lo + int(burst * fs))
    window = (max(0, chirp_lo), max(chirp_lo + 1, chirp_hi))

    bed = noise_bed(cond.noise, wet.size, fs, rng)
    wet = mix_at_snr(wet, bed, fs, cond.snr_db, window)
    wet = wet + rng.standard_normal(wet.size) * 10 ** (cond.preamp_db / 20.0)

    peak = float(np.max(np.abs(wet)))
    if peak > 0:
        wet = wet * (10 ** (cond.peak_dbfs / 20.0) / peak)

    if cond.use_agc:
        wet = agc(wet, fs)
    if cond.clip_ceiling is not None:
        wet = clip(wet, cond.clip_ceiling, soft=True)
    wet = np.clip(wet, -1.0, 1.0)
    wet = quantise(wet, cond.bits)

    true_time = sync_offset + direct_samples / fs
    if cond.ppm:
        wet = clock_offset(wet, cond.ppm)
        true_time *= (1.0 + cond.ppm * 1e-6)
    if cond.codec:
        wet = codec_roundtrip(wet, fs, cond.codec[0], cond.codec[1])

    return Capture(audio=wet, fs=fs, true_time=true_time, condition=cond)
