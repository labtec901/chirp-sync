"""Exports for editors.

Four formats, deliberately ordered by how much you can rely on them:

``csv`` / ``json``
    The offsets themselves, exactly as measured.  Nothing can misinterpret
    these, they diff cleanly, and they are what to script against.
``fcpxml``
    One project per take with the clips stacked on lanes and offset to match.
    Imports into DaVinci Resolve and Premiere.  Video timelines quantise to
    frames, so an NLE will round these -- the CSV keeps the sub-millisecond
    truth for audio work.
``edl``
    CMX3600, one file per take.  Ancient and universally readable, but it only
    describes one video track, so it is a fallback rather than a target.
"""

from __future__ import annotations

import csv
import html
import json
from fractions import Fraction
from pathlib import Path

from .takes import Take

FCPXML_TIMEBASE = 48000  # rational denominator; sample-accurate at 48 kHz


def _rational(seconds: float, timebase: int = FCPXML_TIMEBASE) -> str:
    """FCPXML time value, e.g. ``"144000/48000s"``."""
    n = int(round(seconds * timebase))
    if n == 0:
        return "0s"
    frac = Fraction(n, timebase)
    if frac.denominator == 1:
        return f"{frac.numerator}s"
    return f"{frac.numerator}/{frac.denominator}s"


def _timecode(seconds: float, fps: float) -> str:
    if fps <= 0:
        fps = 25.0
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds * fps))
    f = total % int(round(fps))
    total //= int(round(fps))
    return f"{total // 3600:02d}:{(total // 60) % 60:02d}:{total % 60:02d}:{f:02d}"


# --- CSV / JSON --------------------------------------------------------------


def write_csv(takes: list[Take], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "take", "clip", "file",
            "sync_time_s", "timeline_offset_s", "timeline_offset_ms",
            "duration_s", "score", "clarity_db", "direct_ratio",
        ])
        for take in takes:
            offs = take.offsets()
            for clip in take.clips:
                d = clip.as_dict()
                w.writerow([
                    take.take, clip.path.name, str(clip.path),
                    f"{clip.time:.6f}", f"{offs[clip.path]:.6f}",
                    f"{offs[clip.path] * 1000:.3f}", f"{clip.duration:.3f}",
                    d["score"], d["clarity_db"], d["direct_ratio"],
                ])
    return path


def write_json(takes: list[Take], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "takes": [t.as_dict() for t in takes]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --- FCPXML ------------------------------------------------------------------


def _fmt_id(fps: float, width: int, height: int) -> str:
    return f"fmt{int(round(fps * 1000))}_{width}x{height}"


def write_fcpxml(takes: list[Take], path: str | Path,
                 project_name: str = "chirp-sync") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    assets: dict[str, tuple[str, object]] = {}
    formats: dict[str, tuple[float, int, int]] = {}

    def asset_for(clip) -> str:
        key = str(clip.path)
        if key in assets:
            return assets[key][0]
        aid = f"a{len(assets) + 1}"
        assets[key] = (aid, clip)
        return aid

    for take in takes:
        for clip in take.clips:
            asset_for(clip)
            if clip.has_video and clip.frame_rate:
                w = clip.width or 1920
                h = clip.height or 1080
                formats.setdefault(_fmt_id(clip.frame_rate, w, h),
                                   (clip.frame_rate, w, h))
    if not formats:
        formats["fmtdefault"] = (25.0, 1920, 1080)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<!DOCTYPE fcpxml>',
             '<fcpxml version="1.9">',
             '  <resources>']

    for fid, (fps, w, h) in formats.items():
        frac = Fraction(1 / fps).limit_denominator(120000)
        lines.append(
            f'    <format id="{fid}" name="chirpsync{int(round(fps))}" '
            f'frameDuration="{frac.numerator}/{frac.denominator}s" '
            f'width="{w}" height="{h}" colorSpace="1-1-1 (Rec. 709)"/>')

    default_fmt = next(iter(formats))
    for key, (aid, clip) in assets.items():
        fid = default_fmt
        if clip.has_video and clip.frame_rate:
            cand = _fmt_id(clip.frame_rate, clip.width or 1920, clip.height or 1080)
            if cand in formats:
                fid = cand
        uri = Path(key).resolve().as_uri()
        dur = _rational(clip.duration or 1.0)
        lines.append(
            f'    <asset id="{aid}" name="{html.escape(clip.path.stem)}" '
            f'start="0s" duration="{dur}" '
            f'hasVideo="{1 if clip.has_video else 0}" hasAudio="1" '
            f'format="{fid}" audioSources="1" audioChannels="2" audioRate="48000">')
        lines.append(f'      <media-rep kind="original-media" src="{html.escape(uri)}"/>')
        lines.append('    </asset>')

    lines.append('  </resources>')
    lines.append('  <library>')
    lines.append(f'    <event name="{html.escape(project_name)}">')

    for take in takes:
        if not take.clips:
            continue
        offs = take.offsets()
        # The clip that heard the chirp latest starts earliest on the timeline,
        # so it anchors the spine and every other clip hangs off it at a
        # non-negative offset.
        anchor = max(take.clips, key=lambda c: c.time)
        others = [c for c in take.clips if c is not anchor]
        total = max((offs[c.path] + (c.duration or 0.0)) for c in take.clips)
        name = html.escape(take.take)

        lines.append(f'      <project name="{name}">')
        lines.append(f'        <sequence format="{default_fmt}" '
                     f'duration="{_rational(total)}" tcStart="0s" tcFormat="NDF" '
                     f'audioLayout="stereo" audioRate="48kHz">')
        lines.append('          <spine>')
        a_id = assets[str(anchor.path)][0]
        lines.append(
            f'            <asset-clip ref="{a_id}" offset="{_rational(offs[anchor.path])}" '
            f'name="{html.escape(anchor.path.stem)}" start="0s" '
            f'duration="{_rational(anchor.duration or 1.0)}" format="{default_fmt}">')
        for lane, clip in enumerate(others, start=1):
            rel = offs[clip.path] - offs[anchor.path]
            cid = assets[str(clip.path)][0]
            lines.append(
                f'              <asset-clip ref="{cid}" lane="{lane}" '
                f'offset="{_rational(rel)}" name="{html.escape(clip.path.stem)}" '
                f'start="0s" duration="{_rational(clip.duration or 1.0)}" '
                f'format="{default_fmt}"/>')
        lines.append('            </asset-clip>')
        lines.append('          </spine>')
        lines.append('        </sequence>')
        lines.append('      </project>')

    lines.append('    </event>')
    lines.append('  </library>')
    lines.append('</fcpxml>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- EDL ---------------------------------------------------------------------


def write_edls(takes: list[Take], folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for take in takes:
        offs = take.offsets()
        fps = next((c.frame_rate for c in take.clips if c.frame_rate), 25.0)
        title = take.take
        rows = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
        for i, clip in enumerate(take.clips, start=1):
            start = offs[clip.path]
            end = start + (clip.duration or 1.0)
            rows.append(
                f"{i:03d}  AX       AA/V  C        "
                f"{_timecode(0, fps)} {_timecode(clip.duration or 1.0, fps)} "
                f"{_timecode(start, fps)} {_timecode(end, fps)}")
            rows.append(f"* FROM CLIP NAME: {clip.path.name}")
            rows.append(f"* CHIRP SYNC OFFSET: {start * 1000:.3f} ms")
            rows.append("")
        out = folder / f"{take.take}.edl"
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        written.append(out)
    return written
