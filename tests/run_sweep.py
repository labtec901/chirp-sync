"""Robustness sweep: how often does a chirp survive, and how well does it sync?

Two metrics, because they answer different questions.

**Decode rate** -- did the take ID come back at all?  A miss means the clip has
to be grouped by hand, so this is the number that decides whether the tool is
useful.

**Relative sync error** -- given two cameras that both heard the same chirp,
how far apart are their reported sync instants?  This is the number that decides
whether the cut looks right, and it is deliberately *not* the absolute error:
the speaker's own group delay is common to every camera in the room and cancels,
so measuring each clip against the acoustic ground truth would charge the
detector for a bias that never affects an edit.

Run ``python tests/run_sweep.py --quick`` for a short version.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import simulate as S  # noqa: E402

from chirpsync import css, detect, generate  # noqa: E402

FS = 48000
SWEEP_TAKE_ID = 0x0123456789


def _base_conditions() -> list[S.Condition]:
    """A spread of situations a hobbyist shoot actually produces."""
    out = [
        S.Condition("livingroom_1m", rt60=0.35, distance=1.0, noise="hvac", snr_db=25),
        S.Condition("livingroom_3m", rt60=0.35, distance=3.0, noise="hvac", snr_db=15),
        S.Condition("office_5m", rt60=0.55, distance=5.0, noise="hvac", snr_db=10),
        S.Condition("kitchen_hard", rt60=0.75, distance=4.0, room=(8.0, 6.0, 3.0),
                    noise="street", snr_db=8),
        S.Condition("studio_close", rt60=0.20, distance=1.5, mic="shotgun",
                    speaker="studio", noise="pink", snr_db=30),
        S.Condition("hall_8m", rt60=1.2, distance=8.0, room=(15.0, 12.0, 6.0),
                    noise="hvac", snr_db=10),
        S.Condition("gym_12m", rt60=2.2, distance=12.0, room=(25.0, 20.0, 9.0),
                    noise="hvac", snr_db=8),
        S.Condition("cafe_babble", rt60=0.5, distance=3.0, noise="babble", snr_db=0),
        S.Condition("party_music", rt60=0.5, distance=3.0, noise="music", snr_db=-3),
        S.Condition("street_far", rt60=0.6, distance=6.0, mic="action_cam",
                    noise="street", snr_db=3),
        S.Condition("phone_in_pocket", rt60=0.45, distance=2.0, mic="cheap_usb",
                    speaker="phone", drive=0.8, noise="babble", snr_db=5),
        S.Condition("agc_pumping", rt60=0.4, distance=2.5, noise="street", snr_db=8,
                    use_agc=True),
        S.Condition("hot_and_clipped", rt60=0.4, distance=1.0, noise="pink",
                    snr_db=20, clip_ceiling=0.35),
        S.Condition("aac_64k", rt60=0.4, distance=2.5, noise="pink", snr_db=12,
                    codec=("aac", "64k")),
        S.Condition("mp3_128k", rt60=0.4, distance=2.5, noise="pink", snr_db=12,
                    codec=("mp3", "128k")),
        S.Condition("opus_48k", rt60=0.4, distance=2.5, noise="pink", snr_db=12,
                    codec=("opus", "48k")),
        S.Condition("clock_drift", rt60=0.4, distance=2.5, noise="pink", snr_db=15,
                    ppm=300),
        S.Condition("8bit_lofi", rt60=0.4, distance=2.5, noise="pink", snr_db=15,
                    bits=8),
        S.Condition("everything_wrong", rt60=0.9, distance=6.0, mic="action_cam",
                    speaker="laptop", noise="babble", snr_db=2, use_agc=True,
                    clip_ceiling=0.6, codec=("aac", "96k"), ppm=100),
    ]
    return out


def _snr_sweep(noise: str, snrs) -> list[S.Condition]:
    return [S.Condition(f"{noise}_{s:+d}dB", rt60=0.45, distance=3.0, noise=noise,
                        snr_db=s) for s in snrs]


def _reverb_sweep() -> list[S.Condition]:
    specs = [(0.3, 2.0, (6, 5, 3)), (0.6, 4.0, (10, 8, 4)), (1.0, 6.0, (14, 11, 5)),
             (1.5, 9.0, (18, 14, 7)), (2.2, 12.0, (25, 20, 9)), (3.0, 15.0, (30, 24, 11))]
    return [S.Condition(f"rt{rt:.1f}_{d:.0f}m", rt60=rt, distance=d,
                        room=tuple(float(v) for v in room), noise="hvac", snr_db=12)
            for rt, d, room in specs]


def run_case(gen, cond: S.Condition, profile: str, seed: int) -> dict:
    cap = S.simulate(gen.audio, FS, gen.sync_offsets[0], cond,
                     burst_duration=gen.layout["duration"], seed=seed)
    t0 = time.time()
    dets = detect(cap.audio, cap.fs, profiles=[profile])
    elapsed = time.time() - t0
    hit = next((d for d in dets if d.take_id == gen.payload.take_id), None)
    return {
        "ok": hit is not None,
        "abs_err": (hit.time - cap.true_time) if hit else None,
        "score": hit.score if hit else 0.0,
        "false": len([d for d in dets if d.take_id != gen.payload.take_id]),
        "elapsed": elapsed,
    }


def run_pair(gen, cond: S.Condition, profile: str, seed: int) -> float | None:
    """Two cameras, different spots and different mics, same chirp in the air."""
    a = S.simulate(gen.audio, FS, gen.sync_offsets[0],
                   replace(cond, mic="camera_internal", distance=cond.distance),
                   burst_duration=gen.layout["duration"], seed=seed)
    b = S.simulate(gen.audio, FS, gen.sync_offsets[0],
                   replace(cond, mic="lavalier",
                           distance=max(0.8, cond.distance * 0.55)),
                   burst_duration=gen.layout["duration"], seed=seed + 977)
    da = detect(a.audio, a.fs, profiles=[profile])
    db = detect(b.audio, b.fs, profiles=[profile])
    ha = next((d for d in da if d.take_id == gen.payload.take_id), None)
    hb = next((d for d in db if d.take_id == gen.payload.take_id), None)
    if not ha or not hb:
        return None
    return (ha.time - hb.time) - (a.true_time - b.true_time)


def summarise(rows: list[tuple[str, dict]]) -> None:
    width = max(len(name) for name, _ in rows) + 1
    print(f"{'condition':<{width}} {'decode':>7} {'abs err':>10} "
          f"{'score':>7} {'false':>6}")
    print("-" * (width + 33))
    for name, r in rows:
        err = f"{r['abs_err'] * 1000:+.2f} ms" if r["abs_err"] is not None else "  --"
        decoded = r.get("partial", "PASS" if r["ok"] else "FAIL")
        print(f"{name:<{width}} {decoded:>7} "
              f"{err:>10} {r['score']:>7.3f} {r['false']:>6}")
    ok = sum(r["ok"] for _, r in rows)
    fp = sum(r["false"] for _, r in rows)
    print("-" * (width + 33))
    print(f"decoded {ok}/{len(rows)}   false positives {fp}")


def sensitivity(profile: str, trials: int = 5) -> list[tuple[int, int]]:
    """Decode rate against pure AWGN, with no room and no transducers.

    This separates the modem from everything around it.  Composite conditions
    fail far earlier, and it is worth knowing whether that is the modem running
    out of margin or the room stealing the direct path -- this says which.
    """
    rng = np.random.default_rng(0)
    gen = generate(profile=profile, take_id=SWEEP_TAKE_ID, sample_rate=FS,
                   lead_in=0.5, lead_out=0.5)
    lo = int(0.5 * FS)
    hi = int((0.5 + gen.layout["duration"]) * FS)
    level = S.band_rms(gen.audio[lo:hi], FS)
    out = []
    for snr in range(0, -31, -3):
        ok = 0
        for _ in range(trials):
            noise = rng.standard_normal(gen.audio.size)
            noise *= (level / (10 ** (snr / 20.0))) / S.band_rms(noise, FS)
            dets = detect(gen.audio + noise, FS, profiles=[profile])
            ok += any(d.take_id == gen.payload.take_id for d in dets)
        out.append((snr, ok))
        if ok == 0:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default="balanced",
                    help="comma separated, or 'all'")
    ap.add_argument("--trials", type=int, default=3,
                    help="random seeds per condition")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    names = list(css.PROFILES) if args.profiles == "all" else args.profiles.split(",")
    conditions = _base_conditions()
    if not args.quick:
        conditions += _reverb_sweep()
        conditions += _snr_sweep("babble", range(10, -21, -5))
        conditions += _snr_sweep("music", range(10, -21, -5))
        conditions += _snr_sweep("white", range(0, -31, -5))
    trials = 1 if args.quick else args.trials

    overall: dict[str, tuple[int, int]] = {}
    for pname in names:
        gen = generate(profile=pname, take_id=SWEEP_TAKE_ID, sample_rate=FS,
                       lead_in=0.6, lead_out=1.5)
        print(f"\n{'=' * 78}\nprofile '{pname}'  "
              f"burst {gen.layout['duration']:.2f}s "
              f"(take ID only)\n{'=' * 78}")

        rows = []
        for cond in conditions:
            acc = {"ok": 0, "false": 0, "score": 0.0}
            errs, times = [], []
            for t in range(trials):
                r = run_case(gen, cond, pname, seed=1000 + 37 * t)
                acc["ok"] += r["ok"]
                acc["false"] += r["false"]
                acc["score"] += r["score"]
                times.append(r["elapsed"])
                if r["abs_err"] is not None:
                    errs.append(r["abs_err"])
            rows.append((cond.name, {
                "ok": acc["ok"] == trials,
                "abs_err": float(np.mean(errs)) if errs else None,
                "score": acc["score"] / trials,
                "false": acc["false"],
                "partial": f"{acc['ok']}/{trials}",
            }))
        summarise(rows)

        print("\nrelative sync between two cameras (the number that matters):")
        pair_errs = []
        for cond in conditions:
            for t in range(trials):
                e = run_pair(gen, cond, pname, seed=2000 + 37 * t)
                if e is not None:
                    pair_errs.append(abs(e) * 1000)
        if pair_errs:
            arr = np.array(pair_errs)
            print(f"  n={arr.size}  median={np.median(arr):.3f} ms  "
                  f"p90={np.percentile(arr, 90):.3f} ms  "
                  f"p99={np.percentile(arr, 99):.3f} ms  max={arr.max():.3f} ms")
            for limit, label in ((1.0, "1 ms"), (1000 / 48, "one frame @48fps"),
                                 (1000 / 24, "one frame @24fps")):
                print(f"    within {label:<18} {100 * np.mean(arr <= limit):5.1f}%")
        if not args.quick:
            curve = sensitivity(pname, trials=max(3, trials))
            floor = min((s for s, k in curve if k > 0), default=0)
            print("\nmodem alone (pure AWGN, no room, no transducers):")
            print("  " + "  ".join(f"{s:+d}dB:{k}" for s, k in curve))
            print(f"  usable down to about {floor:+d} dB in-band SNR")

        ok = sum(r["ok"] for _, r in rows)
        overall[pname] = (ok, len(rows))

    print(f"\n{'=' * 78}")
    for pname, (ok, total) in overall.items():
        print(f"  {pname:<10} {ok}/{total} conditions decoded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
