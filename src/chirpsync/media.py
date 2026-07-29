"""Reading audio out of whatever the cameras produced.

Everything goes through ffmpeg, so a folder can mix MP4 from one camera, MOV
from another and WAV from a field recorder without the rest of the code caring.
Audio is pulled straight to a pipe as mono float at the analysis rate: no temp
files, and no decoding of the video stream we do not need.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .css import FS_WORK

MEDIA_SUFFIXES = {
    # video
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm",
    ".insv", ".lrv", ".3gp", ".mpg", ".mpeg", ".wmv",
    # audio
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aif", ".aiff",
    ".wma", ".caf",
}


class MediaError(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffprobe(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True, timeout=120).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return {}


@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    frame_rate: float = 0.0
    width: int = 0
    height: int = 0
    audio_rate: int = 0
    audio_channels: int = 0

    @property
    def name(self) -> str:
        return self.path.name


def _parse_rate(text: str | None) -> float:
    if not text:
        return 0.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    info = MediaInfo(path=path)
    data = _ffprobe(path)
    fmt = data.get("format", {})
    try:
        info.duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        info.duration = 0.0
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and not info.has_video:
            # Cover art is a "video" stream too; a real one has a frame rate.
            rate = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            if stream.get("disposition", {}).get("attached_pic"):
                continue
            info.has_video = True
            info.frame_rate = rate
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_rate = int(stream.get("sample_rate") or 0)
            info.audio_channels = int(stream.get("channels") or 0)
    if not info.duration and info.path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                info.duration = w.getnframes() / float(w.getframerate())
                info.has_audio = True
                info.audio_rate = w.getframerate()
                info.audio_channels = w.getnchannels()
        except (OSError, wave.Error):
            pass
    return info


def load_audio(path: str | Path, rate: int = FS_WORK) -> tuple[np.ndarray, int]:
    """Decode a file's first audio stream to mono float at ``rate``."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"no such file: {path}")
    if not have_ffmpeg():
        return _load_wav_fallback(path)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        hint = detail[-1] if detail else "no audio stream"
        raise MediaError(f"could not read audio from {path.name}: {hint}")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    return audio, rate


def _load_wav_fallback(path: Path) -> tuple[np.ndarray, int]:
    """Enough of a WAV reader to work without ffmpeg installed."""
    if path.suffix.lower() != ".wav":
        raise MediaError(f"ffmpeg is required to read {path.suffix} files")
    with wave.open(str(path), "rb") as w:
        channels, width, fs, frames = (w.getnchannels(), w.getsampwidth(),
                                       w.getframerate(), w.getnframes())
        raw = w.readframes(frames)
    dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
    if dtype is None:
        raise MediaError(f"unsupported WAV sample width in {path.name}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(1 << (8 * width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, fs


def write_wav(path: str | Path, audio: np.ndarray, rate: int,
              subtype: str = "pcm16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    if subtype == "float32":
        payload = audio.astype("<f4").tobytes()
        _write_wav_raw(path, payload, rate, channels=1, bits=32, fmt=3)
        return
    data = np.round(audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(data.tobytes())


def _write_wav_raw(path: Path, payload: bytes, rate: int, channels: int,
                   bits: int, fmt: int) -> None:
    block = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, fmt, channels, rate,
                          rate * block, block, bits)
    header += b"data" + struct.pack("<I", len(payload))
    path.write_bytes(header + payload)


def find_media(paths: list[str | Path], recursive: bool = True) -> list[Path]:
    """Expand files and folders into a sorted list of media files."""
    out: list[Path] = []
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(f for f in it
                       if f.is_file() and f.suffix.lower() in MEDIA_SUFFIXES
                       and not f.name.startswith("."))
        elif p.is_file():
            out.append(p)
    return sorted(set(out), key=lambda f: (str(f.parent), f.name))
