"""Fabricate a realistic multi-camera shoot as actual video files.

Three takes, each heard by several cameras in different spots with different
mics, codecs and clock errors, plus a couple of decoys: a camera that rolled
through two takes without stopping, and a clip that never heard a chirp at all.

The point is to exercise the real pipeline end to end -- ffmpeg demux, detection,
grouping, export -- on files an editor could actually open, and to check the
reported offsets against the offsets we know we baked in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import simulate as S  # noqa: E402

from chirpsync import generate  # noqa: E402
from chirpsync.media import write_wav  # noqa: E402

FS = 48000


@dataclass
class Camera:
    name: str
    mic: str
    distance: float
    codec: tuple[str, str] | None
    fps: float
    ppm: float = 0.0
    use_agc: bool = False
    # Seconds of recording before the chirp: this is the ground-truth stagger
    # between cameras that the parser has to recover.
    preroll: float = 2.0


CAMERAS = [
    Camera("A_CAM_MAIN", "shotgun", 1.8, ("aac", "128k"), 24.0, preroll=3.10),
    Camera("B_CAM_WIDE", "camera_internal", 4.2, ("aac", "96k"), 24.0, ppm=45,
           use_agc=True, preroll=1.35),
    Camera("C_CAM_POCKET", "action_cam", 6.0, ("aac", "64k"), 30.0, ppm=-70,
           preroll=5.80),
    Camera("Z_AUDIO_REC", "lavalier", 2.4, None, 0.0, preroll=0.45),
]

TAKES = [
    ("Scene 1 / Take 1", S.Condition("room", rt60=0.42, noise="hvac", snr_db=18)),
    ("Scene 1 / Take 2", S.Condition("room", rt60=0.42, noise="babble", snr_db=8)),
    ("Scene 2 / Take 1", S.Condition("room", rt60=0.42, noise="street", snr_db=12)),
]


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])


def mux(out: Path, audio: np.ndarray, fps: float, tag: str) -> None:
    """Wrap audio in a real container, with a video track when fps > 0."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = out.with_suffix(".tmp.wav")
    write_wav(tmp_wav, audio, FS)
    dur = len(audio) / FS
    if fps > 0:
        run(["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i",
             f"testsrc=size=640x360:rate={fps:g}:duration={dur:.3f}",
             "-i", str(tmp_wav),
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-shortest",
             "-metadata", f"comment={tag}", str(out)])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav),
             "-c:a", "aac", "-b:a", "192k", str(out)])
    tmp_wav.unlink(missing_ok=True)


def build(dest: Path, profile: str) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    truth: list[dict] = []

    chirps = []
    for label, _cond in TAKES:
        chirp = generate(profile=profile, sample_rate=FS,
                         lead_in=0.0, lead_out=0.0)
        chirps.append(chirp)

    for take_index, ((label, cond), chirp) in enumerate(zip(TAKES, chirps), start=1):
        for cam in CAMERAS:
            # Pad so the chirp lands `preroll` seconds into this camera's clip,
            # with tail afterwards.
            pre = np.zeros(int(cam.preroll * FS))
            post = np.zeros(int(3.0 * FS))
            source = np.concatenate([pre, chirp.audio, post])

            cond_cam = S.Condition(
                name=cam.name, rt60=cond.rt60, distance=cam.distance,
                noise=cond.noise, snr_db=cond.snr_db, mic=cam.mic,
                speaker="phone", drive=0.35, use_agc=cam.use_agc,
                codec=cam.codec, ppm=cam.ppm, seed=take_index * 31 + len(cam.name),
            )
            cap = S.simulate(source, FS, cam.preroll, cond_cam,
                             burst_duration=chirp.layout["duration"])
            name = f"{cam.name}_T{take_index:02d}"
            ext = ".mp4" if cam.fps > 0 else ".m4a"
            out = dest / f"{name}{ext}"
            mux(out, cap.audio, cam.fps, f"{label} / {cam.name}")
            truth.append({
                "file": out.name, "take": chirp.payload.take, "label": label,
                "camera": cam.name, "true_sync_time": cap.true_time,
            })

    # A camera left rolling straight through takes 1 and 2.
    gap = np.zeros(int(4.0 * FS))
    rolling_src = np.concatenate([
        np.zeros(int(1.2 * FS)), chirps[0].audio, gap, chirps[1].audio,
        np.zeros(int(2.0 * FS))])
    cond_roll = S.Condition("rolling", rt60=0.42, distance=3.0, noise="hvac",
                            snr_db=15, mic="camera_internal", codec=("aac", "128k"))
    cap = S.simulate(rolling_src, FS, 1.2, cond_roll,
                     burst_duration=chirps[0].layout["duration"])
    mux(dest / "D_CAM_ROLLING.mp4", cap.audio, 25.0, "rolled through two takes")
    lead = cap.true_time - 1.2
    truth.append({"file": "D_CAM_ROLLING.mp4", "take": chirps[0].payload.take,
                  "label": TAKES[0][0], "camera": "D_CAM_ROLLING",
                  "true_sync_time": cap.true_time})
    second = 1.2 + chirps[0].duration + 4.0 + lead
    truth.append({"file": "D_CAM_ROLLING.mp4", "take": chirps[1].payload.take,
                  "label": TAKES[1][0], "camera": "D_CAM_ROLLING",
                  "true_sync_time": second})

    # And a clip that simply never heard one.
    rng = np.random.default_rng(7)
    silence = S.noise_bed("babble", int(8.0 * FS), FS, rng) * 0.05
    mux(dest / "E_CAM_NOCHIRP.mp4", silence, 24.0, "no chirp here")

    (dest / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return {"files": len(truth), "takes": len(TAKES)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", nargs="?", default="artifacts/demo_shoot")
    ap.add_argument("-p", "--profile", default="balanced")
    args = ap.parse_args()
    dest = Path(args.dest)
    info = build(dest, args.profile)
    print(f"wrote {info['takes']} takes into {dest}")
    for f in sorted(dest.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
