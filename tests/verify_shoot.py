"""Check a generated demo shoot against the offsets that were baked into it.

Reports two things per take: whether every clip that heard the chirp was found
and grouped correctly, and how far each clip's recovered timeline offset is from
the truth once the take's common bias is removed.  Removing the common bias is
the honest comparison -- a constant shift applies to every clip in a take and so
moves the whole group together, leaving the edit correct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chirpsync import detect  # noqa: E402
from chirpsync.media import find_media, load_audio, probe  # noqa: E402
from chirpsync.takes import ClipDetection, build_takes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default="artifacts/demo_shoot")
    ap.add_argument("-p", "--profile", default="auto")
    args = ap.parse_args()

    folder = Path(args.folder)
    truth = json.loads((folder / "ground_truth.json").read_text())
    expected = defaultdict(dict)          # take -> {filename: true_sync_time}
    for row in truth:
        expected[row["take"]][row["file"]] = row["true_sync_time"]
    labels = {row["take"]: row["label"] for row in truth}

    profiles = None if args.profile == "auto" else [args.profile]
    clips = []
    for path in find_media([folder]):
        try:
            audio, rate = load_audio(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"skip {path.name}: {exc}")
            continue
        info = probe(path)
        for det in detect(audio, rate, profiles=profiles):
            clips.append(ClipDetection(
                path=path, take=det.take, take_id=det.take_id, time=det.time,
                score=det.score,
                clarity_db=det.clarity_db, direct_ratio=det.direct_ratio,
                duration=info.duration, has_video=info.has_video,
                frame_rate=info.frame_rate, width=info.width, height=info.height))

    takes = build_takes(clips)
    found_keys = {(t.take, c.path.name) for t in takes for c in t.clips}
    want_keys = {(tk, f) for tk, files in expected.items() for f in files}

    missing = sorted(want_keys - found_keys)
    spurious = sorted(found_keys - want_keys)

    all_err = []
    print(f"{'take':<10} {'clip':<24} {'measured':>11} {'truth':>11} {'error':>10}")
    print("-" * 70)
    for take in takes:
        want = expected.get(take.take, {})
        if not want:
            continue
        offs = take.offsets()
        latest_true = max(want.values())
        rows = []
        for clip in take.clips:
            t = want.get(clip.path.name)
            if t is None:
                continue
            rows.append((clip.path.name, offs[clip.path], latest_true - t))
        bias = np.median([m - e for _, m, e in rows]) if rows else 0.0
        for name, measured, exp in rows:
            err = (measured - bias - exp) * 1000
            all_err.append(abs(err))
            print(f"{take.take:<10} {name:<24} {measured * 1000:>9.2f}ms "
                  f"{exp * 1000:>9.2f}ms {err:>8.3f}ms")
        print(f"{'':<10} {'(take label ' + labels.get(take.take, '?') + ')':<24}"
              f"  common bias {bias * 1000:+.3f} ms")
        print("-" * 70)

    ok = True
    if missing:
        ok = False
        print("\nMISSED (clip heard the chirp but was not decoded):")
        for tk, f in missing:
            print(f"  {tk}  {f}")
    if spurious:
        ok = False
        print("\nUNEXPECTED detections:")
        for tk, f in spurious:
            print(f"  {tk}  {f}")

    if all_err:
        a = np.array(all_err)
        print("\nsync error after removing each take's common bias:")
        print(f"  n={a.size}  median={np.median(a):.3f} ms  "
              f"p90={np.percentile(a, 90):.3f} ms  max={a.max():.3f} ms")
        for limit, label in ((0.1, "0.1 ms"), (1.0, "1 ms"),
                             (1000 / 48, "one frame @48fps")):
            print(f"    within {label:<18} {100 * np.mean(a <= limit):5.1f}%")
    print(f"\nrecovered {len(found_keys & want_keys)}/{len(want_keys)} expected detections")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
