# chirp-sync

[![CI](https://github.com/labtec901/chirp-sync/actions/workflows/ci.yml/badge.svg)](https://github.com/labtec901/chirp-sync/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chirp-sync.svg)](https://pypi.org/project/chirp-sync/)
[![Python](https://img.shields.io/pypi/pyversions/chirp-sync.svg)](https://pypi.org/project/chirp-sync/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An acoustic slate for grouping and synchronizing multi-camera recordings. Roll
every camera, play one short chirp, and recover the same take ID and precise
acoustic sync point from every recording that heard it.

- **Web generator:** <https://labtec901.github.io/chirp-sync/>
- **Python package:** <https://pypi.org/project/chirp-sync/>
- **Source:** <https://github.com/labtec901/chirp-sync>

The acoustic payload is intentionally fixed: every chirp carries one random
40-bit take ID.

## Installation

The CLI supports Python 3.10 and newer. Installing with
[pipx](https://pipx.pypa.io/) keeps command-line applications isolated:

```bash
pipx install chirp-sync
```

Installing with pip is also supported:

```bash
python -m pip install chirp-sync
```

[FFmpeg](https://ffmpeg.org/download.html) must be available on `PATH` to scan
MP4, MOV, MP3, M4A, and other camera formats. A built-in WAV reader covers WAV
files. The microphone listener is optional:

```bash
python -m pip install "chirp-sync[mic]"
```

## Typical workflow

1. Open the [web generator](https://labtec901.github.io/chirp-sync/) on a phone.
2. Start every camera and audio recorder.
3. Press **Play chirp** before the take.
4. Scan the recordings afterward:

```bash
chirp-sync scan /path/to/recordings
```

Files with the same ID are grouped together and given offsets on a shared
timeline. The chirp arrival is measured from each recording's own audio.

### Prefix decoded IDs onto filenames

Preview the in-place rename first:

```bash
chirp-sync scan /path/to/recordings --prepend-id --dry-run
```

Then apply it:

```bash
chirp-sync scan /path/to/recordings --prepend-id
```

For example:

```text
camera-a.mov  ->  ZY41WN7M_camera-a.mov
camera-b.mp4  ->  ZY41WN7M_camera-b.mp4
```

Name collisions receive a numeric suffix, already-prefixed files are treated as
complete, and recordings containing multiple take IDs are flagged for manual
naming.

### Export or organize a shoot

```bash
# Write every supported timeline/report format.
chirp-sync scan recordings --all-exports --outdir exports

# Copy files into take-ID folders.
chirp-sync scan recordings --organize sorted

# Preview moving instead of copying.
chirp-sync scan recordings --organize sorted --move --dry-run
```

Available exports are CSV, JSON, Final Cut Pro XML, and one CMX3600 EDL per
take.

## CLI reference

### `chirp-sync gen`

Generate a WAV with a fresh random take ID. The fast profile is the default.

```bash
chirp-sync gen
chirp-sync gen --take X4THAPJ9
chirp-sync gen --profile robust --repeats 2 -o stage.wav
```

Options control sample rate, peak level, lead-in, lead-out, repeat count, repeat
gap, and PCM16 or float32 WAV output.

### `chirp-sync scan`

Recursively scan files and directories, group matching IDs, calculate relative
offsets, export timelines, organize copies, or prepend IDs in place.

```bash
chirp-sync scan PATH [PATH ...]
chirp-sync scan recordings --profile fast
chirp-sync scan recordings --json results.json
chirp-sync scan recordings --prepend-id --dry-run
```

Automatic profile detection is the scan default.

### `chirp-sync info`

Inspect one recording in detail:

```bash
chirp-sync info recording.mov
```

The report includes media properties, take ID, chirp arrival time, profile,
detection score, clarity, and direct-path strength.

### `chirp-sync listen`

Record briefly from the desktop microphone and check that a chirp can be
decoded in the room:

```bash
chirp-sync listen --seconds 20
```

Run `chirp-sync COMMAND --help` for every option.

## Profiles

| Profile | SF | Symbol | Data symbols | Complete burst | Intended use |
|---|---:|---:|---:|---:|---|
| `fast` (default) | 7 | 32 ms | 24 | 1.088 s | Normal close-range use |
| `balanced` | 8 | 64 ms | 21 | 1.984 s | More range or reverberation |
| `robust` | 9 | 128 ms | 19 | 3.712 s | Large, noisy, or echoing rooms |

The website adds 0.15 seconds of silence at each end. The CLI defaults to 0.25
seconds at each end. The table reports the encoded burst before that padding.

## Encoding specification

The fixed frame is:

```text
[ 8 unshifted up-chirps ][ 2 unshifted down-chirps ][ encoded ID symbols ]
        preamble                    SFD                 protected data
```

The encoded block begins as exactly seven bytes:

```text
[ five-byte unsigned take ID ][ two-byte CRC-16 ]
```

Integers and bit groups are most-significant-byte and most-significant-bit
first unless stated otherwise.

1. The displayed eight-character Crockford-base32 ID represents a 40-bit
   unsigned integer. The audio representation uses the five raw bytes.
2. Append a big-endian CRC-16/CCITT-FALSE over those bytes. Parameters are
   polynomial `0x1021`, initial value `0xffff`, no reflection, and final XOR
   `0x0000`.
3. XOR all seven bytes with PN9 whitening using `x^9 + x^5 + 1`, initial state
   `0x1ff`, and least-significant-bit-first mask output. The fixed mask is
   `ff e1 1d 9a ed 85 33`.
4. Unpack MSB-first and encode the 56 bits with a zero-terminated,
   constraint-length-7 convolutional code using octal generators
   `(171, 133, 165)`. Six zero tail bits terminate the trellis. The third
   parity output is omitted when `step mod 3 == 2`, producing 166 coded bits.
5. Zero-pad to a whole CSS symbol and apply the deterministic stride
   interleaver. The padded length and stride are `(168, 103)` for fast and
   balanced, and `(171, 106)` for robust.
6. Group into `SF`-bit values `v` and Gray-map each CSS shift as
   `g = v XOR (v >> 1)`.

Each data value selects a cyclic shift of a continuous-phase linear up-chirp in
the 1 to 5 kHz band. With `N = 2^SF`, bandwidth `BW = 4000 Hz`, and
`Ts = N / BW`, the instantaneous frequency is:

```text
f(t) = 1000 + 4000 * ((t / Ts + g / N) mod 1),  0 <= t < Ts
```

The receiver finds the preamble, confirms the down-chirp delimiter, estimates
the room's delay profile, obtains soft bit likelihoods from the CSS symbols,
deinterleaves, runs a 64-state Viterbi decoder, removes whitening, and accepts
the take ID only when its CRC passes.

## Audio files and privacy

`chirp-sync gen` writes 48 kHz mono signed 16-bit PCM WAV by default and
normalizes the peak to -3 dBFS. `--float32` selects 32-bit floating-point PCM.
The protocol supports recordings stored with common lossy camera codecs.

The web generator runs in the browser and stores its session log in browser
storage.

## Python API

```python
from chirpsync import detect, generate

chirp = generate()  # fast profile by default
detections = detect(chirp.audio, chirp.sample_rate)
assert detections[0].take == chirp.payload.take
```

## Development

```bash
git clone https://github.com/labtec901/chirp-sync.git
cd chirp-sync
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python -m pytest
ruff check .
python -m build
python -m twine check dist/*
```

Node.js is required for browser and Python encoder parity tests. FFmpeg is
required for full media and codec simulation tests.

GitHub Actions tests Python 3.10 through 3.13, deploys `webapp/` to GitHub Pages,
and publishes signed distributions to PyPI when a GitHub release is published.
PyPI publication uses trusted publishing rather than a long-lived API token.

## License

[MIT](LICENSE)
