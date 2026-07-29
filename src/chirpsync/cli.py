"""Command line interface.

    chirp-sync gen    -- make a chirp to play on set
    chirp-sync scan   -- find chirps, group, export, organize, or prefix files
    chirp-sync info   -- inspect one file in detail
    chirp-sync listen -- decode from the microphone, for checking a room
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__, codec, css, media, nle, organize
from .detector import DEFAULT_THRESHOLD, detect
from .generator import generate
from .takes import ClipDetection, build_takes


def _eprint(*args) -> None:
    print(*args, file=sys.stderr)


# --- gen ---------------------------------------------------------------------


def cmd_gen(args: argparse.Namespace) -> int:
    take_id = codec.take_id_from_str(args.take) if args.take else None
    chirp = generate(
        profile=args.profile,
        take_id=take_id,
        sample_rate=args.rate,
        level_dbfs=args.level,
        lead_in=args.lead_in,
        lead_out=args.lead_out,
        repeats=args.repeats,
        gap=args.gap,
    )

    out = Path(args.output) if args.output else Path(f"chirp_{chirp.payload.take}.wav")
    media.write_wav(out, chirp.audio, chirp.sample_rate,
                    "float32" if args.float32 else "pcm16")

    layout = chirp.layout
    print(f"take        {chirp.payload.take}")
    print(f"profile     {args.profile} (sf={css.get_profile(args.profile).sf})")
    print(f"chirp       {layout['duration']:.2f} s"
          + (f" x{chirp.repeats}" if chirp.repeats > 1 else ""))
    print(f"file        {out}  ({chirp.duration:.2f} s @ {chirp.sample_rate} Hz)")
    return 0


# --- scan --------------------------------------------------------------------


def _scan_files(paths: list[Path], args: argparse.Namespace) -> list[ClipDetection]:
    profiles = None if args.profile == "auto" else [args.profile]
    found: list[ClipDetection] = []
    for i, path in enumerate(paths, start=1):
        prefix = f"[{i}/{len(paths)}] {path.name}"
        try:
            audio, rate = media.load_audio(path)
        except media.MediaError as exc:
            _eprint(f"{prefix}: skipped ({exc})")
            continue
        started = time.time()
        dets = detect(audio, rate, profiles=profiles, threshold=args.threshold)
        info = media.probe(path)
        for det in dets:
            found.append(ClipDetection(
                path=path, take=det.take, take_id=det.take_id, time=det.time,
                score=det.score,
                clarity_db=det.clarity_db, direct_ratio=det.direct_ratio,
                duration=info.duration or (audio.size / rate),
                has_video=info.has_video, frame_rate=info.frame_rate,
                width=info.width, height=info.height,
            ))
        elapsed = time.time() - started
        if dets:
            takes = ", ".join(sorted({d.take for d in dets}))
            print(f"{prefix}: {len(dets)} chirp(s) -> {takes}  ({elapsed:.1f}s)")
        elif not args.quiet:
            print(f"{prefix}: no chirp found  ({elapsed:.1f}s)")
    return found


def cmd_scan(args: argparse.Namespace) -> int:
    paths = media.find_media(args.paths, recursive=not args.no_recursive)
    if not paths:
        _eprint("no media files found")
        return 1
    if not media.have_ffmpeg():
        _eprint("warning: ffmpeg not found; only WAV files can be read")

    print(f"scanning {len(paths)} file(s)...")
    clips = _scan_files(paths, args)
    if not clips:
        _eprint("\nno chirps decoded in any file.")
        return 2

    takes = build_takes(clips)

    if args.prepend_id:
        moves, ambiguous = organize.plan_prepend_ids(takes)
        if ambiguous:
            _eprint("warning: files containing more than one take ID were not renamed:")
            for path, ids in ambiguous.items():
                _eprint(f"  {path}: {', '.join(ids)}")
        done = organize.apply(moves, move=True, dry_run=args.dry_run)
        verb = "would rename" if args.dry_run else "renamed"
        print(f"\n{verb} {len(done)} file(s) in place")
        for item in done[:20]:
            print(f"  {item.source.name}  ->  {item.target.name}")
        if len(done) > 20:
            print(f"  ... and {len(done) - 20} more")
        if not args.dry_run:
            renamed = {item.source: item.target for item in done}
            for take in takes:
                for clip in take.clips:
                    clip.path = renamed.get(clip.path, clip.path)

    _report(takes)

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    if args.csv or args.all_exports:
        print("csv     ", nle.write_csv(takes, args.csv or outdir / "chirp-sync.csv"))
    if args.json or args.all_exports:
        print("json    ", nle.write_json(takes, args.json or outdir / "chirp-sync.json"))
    if args.fcpxml or args.all_exports:
        print("fcpxml  ", nle.write_fcpxml(takes, args.fcpxml or outdir / "chirp-sync.fcpxml"))
    if args.edl_dir or args.all_exports:
        written = nle.write_edls(takes, args.edl_dir or outdir / "edl")
        print(f"edl      {len(written)} file(s) in {(args.edl_dir or outdir / 'edl')}")

    if args.organize:
        moves = organize.plan(takes, args.organize, rename=not args.no_rename)
        done = organize.apply(moves, move=args.move, dry_run=args.dry_run)
        verb = "would " + ("move" if args.move else "copy") if args.dry_run else \
               ("moved" if args.move else "copied")
        print(f"\n{verb} {len(done)} file(s) into {args.organize}")
        if args.dry_run:
            for item in done[:20]:
                print(f"  {item.source.name}  ->  "
                      f"{item.target.relative_to(Path(args.organize))}")
            if len(done) > 20:
                print(f"  ... and {len(done) - 20} more")
    return 0


def _report(takes) -> None:
    print(f"\n{len(takes)} take(s):\n")
    for take in takes:
        print(f"  {take.take}   {len(take.clips)} clip(s)")
        offs = take.offsets()
        for clip in take.clips:
            soft = "" if clip.direct_ratio >= 0.9 else "  [reverberant]"
            print(f"      {clip.path.name:<42} offset {offs[clip.path] * 1000:+9.2f} ms"
                  f"   clarity {clip.clarity_db:5.1f} dB{soft}")
        if len(take.clips) == 1:
            print("      (only one clip; nothing to sync against)")
        print()


# --- info --------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    path = Path(args.path)
    info = media.probe(path)
    print(f"file        {path}")
    print(f"duration    {info.duration:.3f} s")
    if info.has_video:
        print(f"video       {info.width}x{info.height} @ {info.frame_rate:.3f} fps")
    print(f"audio       {info.audio_rate} Hz, {info.audio_channels} ch")
    audio, rate = media.load_audio(path)
    profiles = None if args.profile == "auto" else [args.profile]
    dets = detect(audio, rate, profiles=profiles, threshold=args.threshold)
    if not dets:
        print("\nno chirp decoded")
        return 2
    print(f"\n{len(dets)} chirp(s):")
    for det in dets:
        d = det.as_dict()
        print(f"\n  take        {d['take']}")
        print(f"  at          {d['time']:.6f} s")
        print(f"  profile     {d['profile']}")
        print(f"  score       {d['score']:.3f}   clarity {d['clarity_db']:.1f} dB "
              f"(direct chirp vs noise + reverb)")
        print(f"  direct/peak {d['direct_ratio']:.3f}"
              + ("   (clean direct path)" if d["direct_ratio"] >= 0.9
                 else "   (reverberant; sync instant is softer)"))
    return 0


# --- listen ------------------------------------------------------------------


def cmd_listen(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        _eprint("listen needs the optional 'sounddevice' package "
                "(pip install sounddevice)")
        return 1
    import numpy as np

    rate = css.FS_WORK
    print(f"listening for {args.seconds:.0f} s at {rate} Hz... play a chirp now")
    buf = sd.rec(int(args.seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(buf).ravel().astype(float)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"captured {audio.size / rate:.1f} s, peak {20 * np.log10(max(peak, 1e-9)):.1f} dBFS")
    dets = detect(audio, rate, threshold=args.threshold)
    if not dets:
        print("no chirp decoded")
        return 2
    for det in dets:
        print(f"  {det.take}  at {det.time:.4f} s  clarity {det.clarity_db:.1f} dB")
    return 0


# --- argument plumbing -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chirp-sync", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="generate a take-ID chirp WAV")
    g.add_argument("-o", "--output", help="output WAV (default chirp_<TAKE>.wav)")
    g.add_argument("-p", "--profile", default=css.DEFAULT_PROFILE,
                   choices=sorted(css.PROFILES),
                   help="fast = shortest, robust = for big reverberant rooms")
    g.add_argument("--take", help="reuse a specific take ID instead of a new one")
    g.add_argument("--rate", type=int, default=48000)
    g.add_argument("--level", type=float, default=-3.0, help="peak level in dBFS")
    g.add_argument("--lead-in", type=float, default=0.25)
    g.add_argument("--lead-out", type=float, default=0.25)
    g.add_argument("--repeats", type=int, default=1,
                   help="emit the burst more than once for a noisy set")
    g.add_argument("--gap", type=float, default=0.25, help="silence between repeats")
    g.add_argument("--float32", action="store_true", help="write 32-bit float WAV")
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("scan", help="find chirps in clips, group takes, export")
    s.add_argument("paths", nargs="+", help="files and/or folders")
    s.add_argument("-p", "--profile", default="auto",
                   choices=["auto", *sorted(css.PROFILES)],
                   help="auto tries every profile (default)")
    s.add_argument("--outdir", help="where to write exports (default: cwd)")
    s.add_argument("--csv", help="write a CSV of every clip and its offset")
    s.add_argument("--json", help="write full results as JSON")
    s.add_argument("--fcpxml", help="write an FCPXML with one project per take")
    s.add_argument("--edl-dir", help="write one CMX3600 EDL per take into this folder")
    s.add_argument("-a", "--all-exports", action="store_true",
                   help="write csv, json, fcpxml and edl with default names")
    destinations = s.add_mutually_exclusive_group()
    destinations.add_argument("--organize", help="copy clips into per-take folders here")
    destinations.add_argument(
        "--prepend-id", action="store_true",
        help="rename files in place as TAKEID_original-name.ext",
    )
    s.add_argument("--move", action="store_true",
                   help="move files when organizing (default: copy)")
    s.add_argument("--no-rename", action="store_true",
                   help="preserve original filenames when organizing")
    s.add_argument("--dry-run", action="store_true",
                   help="preview planned organization or in-place renaming")
    s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    s.add_argument("-q", "--quiet", action="store_true",
                   help="show decoded files only")
    s.set_defaults(func=cmd_scan)

    i = sub.add_parser("info", help="inspect one file in detail")
    i.add_argument("path")
    i.add_argument("-p", "--profile", default="auto",
                   choices=["auto", *sorted(css.PROFILES)])
    i.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    i.set_defaults(func=cmd_info)

    ls = sub.add_parser("listen", help="decode from the microphone (room check)")
    ls.add_argument("-s", "--seconds", type=float, default=12.0)
    ls.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ls.set_defaults(func=cmd_listen)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _eprint("\ninterrupted")
        return 130
    except (media.MediaError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
