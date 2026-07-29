"""Unit tests for the library: framing, the modem, and the pipeline around it."""

from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import simulate as S  # noqa: E402

from chirpsync import codec, css, detect, fec, frame, generate, nle, organize  # noqa: E402
from chirpsync.takes import ClipDetection, build_takes  # noqa: E402


class CodecTest(unittest.TestCase):

    def test_take_id_round_trip(self):
        for _ in range(200):
            value = codec.new_take_id()
            text = codec.take_id_to_str(value)
            self.assertEqual(len(text), 8)
            self.assertEqual(codec.take_id_from_str(text), value)

    def test_take_id_folds_ambiguous_characters(self):
        # Crockford treats I/L as 1 and O as 0, so a mistyped ID still resolves.
        value = codec.take_id_from_str("01234567")
        self.assertEqual(codec.take_id_from_str("OI234567"), value)

    def test_payload_contains_only_take_id_and_crc(self):
        p = codec.Payload(take_id=0x0123456789)
        blob = codec.encode_payload(p)
        self.assertEqual(len(blob), 7)
        self.assertEqual(blob[:5], bytes.fromhex("0123456789"))
        self.assertEqual(codec.decode_payload(blob), p)

    def test_corrupt_payload_is_rejected(self):
        p = codec.Payload(take_id=1)
        blob = bytearray(codec.encode_payload(p))
        for bit in range(len(blob) * 8):
            bad = bytearray(blob)
            bad[bit // 8] ^= 1 << (bit % 8)
            self.assertIsNone(codec.decode_payload(bytes(bad)),
                              f"single bit flip at {bit} slipped past CRC16")

    def test_payload_rejects_extra_application_fields(self):
        with self.assertRaises(TypeError):
            codec.Payload(take_id=1, timestamp=2)


class FecTest(unittest.TestCase):

    def test_viterbi_recovers_clean_bits(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 104).astype(np.int8)
        coded = fec.conv_encode(bits)
        soft = np.where(coded == 0, 5.0, -5.0)
        self.assertTrue(np.array_equal(fec.viterbi_decode(soft, bits.size), bits))

    def test_viterbi_tolerates_errors(self):
        rng = np.random.default_rng(1)
        recovered = 0
        for _ in range(20):
            bits = rng.integers(0, 2, 104).astype(np.int8)
            coded = fec.conv_encode(bits)
            flip = rng.choice(coded.size, int(0.06 * coded.size), replace=False)
            noisy = coded.copy()
            noisy[flip] ^= 1
            soft = np.where(noisy == 0, 5.0, -5.0)
            if np.array_equal(fec.viterbi_decode(soft, bits.size), bits):
                recovered += 1
        self.assertGreaterEqual(recovered, 18, "coding gain looks wrong")

    def test_punctured_code_length(self):
        # 56 payload bits + 6 tail steps, with two base parities per step and
        # the third parity sent on two of every three steps.
        self.assertEqual(fec.coded_length(56), 166)
        self.assertEqual(fec.conv_encode(np.zeros(56, dtype=np.int8)).size, 166)

    def test_interleaver_is_a_permutation(self):
        for n in (7, 64, 220, 221, 1024):
            idx = fec.interleave_indices(n)
            self.assertEqual(sorted(idx.tolist()), list(range(n)))
            v = np.arange(n)
            self.assertTrue(np.array_equal(fec.deinterleave(fec.interleave(v)), v))

    def test_whitening_is_self_inverse(self):
        data = bytes(range(64))
        self.assertEqual(fec.whiten(fec.whiten(data)), data)
        self.assertNotEqual(fec.whiten(data), data)


class ModemTest(unittest.TestCase):

    def test_default_profile_is_fast(self):
        chirp = generate(sample_rate=48000, lead_in=0.0, lead_out=0.0)
        self.assertEqual(css.DEFAULT_PROFILE, "fast")
        self.assertEqual(chirp.layout["profile"], "fast")

    def test_gray_symbol_mapping_round_trip(self):
        rng = np.random.default_rng(3)
        for sf in (6, 7, 8, 9):
            bits = rng.integers(0, 2, sf * 20).astype(np.int8)
            syms = css.bits_to_symbols(bits, sf)
            self.assertTrue(np.all(syms < (1 << sf)))
            self.assertTrue(np.array_equal(css.symbols_to_bits(syms, sf), bits))

    def test_block_round_trip_all_profiles(self):
        rng = np.random.default_rng(5)
        for sf in (6, 7, 8, 9):
            data = bytes(rng.integers(0, 256, 13))
            syms = frame.encode_block(data, sf)
            llrs = np.where(css.symbols_to_bits(syms, sf) == 0, 6.0, -6.0)
            self.assertEqual(frame.decode_block(llrs, len(data), sf), data)

    def test_clean_round_trip_every_profile_and_rate(self):
        for profile in css.PROFILES:
            for rate in (44100, 48000):
                chirp = generate(profile=profile, sample_rate=rate,
                                 lead_in=0.2, lead_out=0.2)
                dets = detect(chirp.audio, rate, profiles=[profile])
                with self.subTest(profile=profile, rate=rate):
                    self.assertEqual(len(dets), 1)
                    d = dets[0]
                    self.assertEqual(d.take_id, chirp.payload.take_id)
                    self.assertLess(abs(d.time - chirp.sync_offsets[0]), 1e-4)

    def test_profile_is_auto_detected(self):
        for profile in css.PROFILES:
            chirp = generate(profile=profile, sample_rate=48000)
            dets = detect(chirp.audio, 48000)
            with self.subTest(profile=profile):
                self.assertEqual(len(dets), 1)
                self.assertEqual(dets[0].profile, profile)

    def test_fixed_payload_chirp_round_trip(self):
        chirp = generate(profile="balanced", sample_rate=48000)
        dets = detect(chirp.audio, 48000, profiles=["balanced"])
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].take_id, chirp.payload.take_id)

    def test_repeats_are_all_found(self):
        chirp = generate(profile="fast", sample_rate=48000,
                         repeats=3, gap=0.3)
        dets = detect(chirp.audio, 48000, profiles=["fast"])
        self.assertEqual(len(dets), 3)
        self.assertEqual({d.take_id for d in dets}, {chirp.payload.take_id})
        self.assertEqual([d.repeat_index for d in dets], [0, 1, 2])
        for det, expected in zip(dets, chirp.sync_offsets):
            self.assertLess(abs(det.time - expected), 1e-3)

    def test_several_takes_in_one_recording(self):
        """A camera left rolling across takes must yield each one separately."""
        parts, expected = [np.zeros(int(0.4 * 48000))], []
        at = 0.4
        for _ in range(3):
            chirp = generate(profile="fast", sample_rate=48000,
                             lead_in=0.0, lead_out=0.0)
            expected.append((chirp.payload.take_id, at))
            parts.append(chirp.audio)
            gap = np.zeros(int(1.5 * 48000))
            parts.append(gap)
            at += chirp.duration + 1.5
        audio = np.concatenate(parts)
        dets = detect(audio, 48000, profiles=["fast"])
        self.assertEqual(len(dets), 3)
        for det, (take_id, when) in zip(dets, expected):
            self.assertEqual(det.take_id, take_id)
            self.assertLess(abs(det.time - when), 1e-3)

    def test_stereo_input_is_accepted(self):
        chirp = generate(profile="fast", sample_rate=48000)
        stereo = np.stack([chirp.audio, chirp.audio * 0.4], axis=1)
        self.assertEqual(len(detect(stereo, 48000, profiles=["fast"])), 1)

    def test_survives_a_realistic_room(self):
        chirp = generate(profile="balanced", sample_rate=48000,
                         lead_in=0.5, lead_out=1.0)
        cond = S.Condition("room", rt60=0.45, distance=3.0, noise="babble",
                           snr_db=8, mic="camera_internal", speaker="phone",
                           codec=("aac", "128k"))
        cap = S.simulate(chirp.audio, 48000, chirp.sync_offsets[0], cond,
                         burst_duration=chirp.layout["duration"])
        dets = detect(cap.audio, cap.fs, profiles=["balanced"])
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].take_id, chirp.payload.take_id)
        self.assertLess(abs(dets[0].time - cap.true_time), 0.002)


class FalsePositiveTest(unittest.TestCase):
    """Nothing but a real chirp may ever produce a take ID."""

    def _assert_silent(self, audio, label):
        self.assertEqual(detect(audio, 48000), [], f"false positive on {label}")

    def test_no_detection_in_noise_or_music(self):
        rng = np.random.default_rng(11)
        n = int(12.0 * 48000)
        for kind in S.NOISE_KINDS:
            self._assert_silent(S.noise_bed(kind, n, 48000, rng) * 0.2, kind)

    def test_no_detection_in_silence_or_dc(self):
        self._assert_silent(np.zeros(int(6.0 * 48000)), "digital silence")
        self._assert_silent(np.full(int(6.0 * 48000), 0.5), "constant DC")

    def test_no_detection_on_sweeps_that_are_not_chirps(self):
        """A bare frequency sweep is the most plausible natural near-miss."""
        t = np.arange(int(8.0 * 48000)) / 48000
        for f0, f1, dur in ((200, 8000, 8.0), (5000, 1000, 8.0), (1000, 5000, 0.5)):
            phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
            self._assert_silent(0.5 * np.sin(phase), f"sweep {f0}->{f1}")

    def test_truncated_chirp_is_not_reported(self):
        """Half a chirp must fail the CRC rather than yield a wrong take."""
        chirp = generate(profile="balanced", sample_rate=48000, lead_in=0.2)
        cut = chirp.audio[: int(0.2 * 48000) + int(0.6 * chirp.layout["duration"] * 48000)]
        for det in detect(cut, 48000, profiles=["balanced"]):
            self.assertEqual(det.take_id, chirp.payload.take_id)


class PipelineTest(unittest.TestCase):

    def _clip(self, name, take, take_id, when, **kw):
        return ClipDetection(path=Path(name), take=take, take_id=take_id, time=when,
                             score=0.5, clarity_db=10.0,
                             direct_ratio=1.0, duration=kw.pop("duration", 30.0), **kw)

    def test_offsets_are_relative_and_non_negative(self):
        clips = [self._clip("a.mp4", "AAAA1111", 1, 5.0),
                 self._clip("b.mp4", "AAAA1111", 1, 2.0),
                 self._clip("c.mp4", "AAAA1111", 1, 9.5)]
        take = build_takes(clips)[0]
        offs = take.offsets()
        self.assertAlmostEqual(offs[Path("c.mp4")], 0.0)
        self.assertAlmostEqual(offs[Path("a.mp4")], 4.5)
        self.assertAlmostEqual(offs[Path("b.mp4")], 7.5)
        self.assertTrue(all(v >= 0 for v in offs.values()))

    def test_organize_plan_avoids_collisions(self):
        clips = [self._clip("x/shot.mp4", "AAAA1111", 1, 1.0),
                 self._clip("y/shot.mp4", "AAAA1111", 1, 2.0)]
        takes = build_takes(clips)
        with tempfile.TemporaryDirectory() as td:
            moves = organize.plan(takes, td, rename=True)
            targets = [m.target for m in moves]
            self.assertEqual(len(set(targets)), len(targets),
                             "two clips were planned onto the same path")

    def test_organize_sanitizes_labels(self):
        self.assertNotIn("/", organize.sanitize("Scene 4 / Take 2"))
        self.assertNotIn(":", organize.sanitize("a:b"))
        self.assertTrue(organize.sanitize("", fallback="fb"))
        self.assertLessEqual(len(organize.sanitize("x" * 500)), 60)

    def test_prepend_id_renames_in_place_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "camera.mov"
            source.write_bytes(b"clip")
            clip = self._clip(str(source), "AAAA1111", 1, 1.0)
            moves, ambiguous = organize.plan_prepend_ids(build_takes([clip]))
            self.assertFalse(ambiguous)
            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0].target.name, "AAAA1111_camera.mov")

            organize.apply(moves, move=True)
            self.assertFalse(source.exists())
            self.assertTrue(moves[0].target.exists())

            clip.path = moves[0].target
            again, ambiguous = organize.plan_prepend_ids(build_takes([clip]))
            self.assertFalse(ambiguous)
            self.assertEqual(again, [])

    def test_prepend_id_skips_files_with_multiple_take_ids(self):
        path = Path("rolling.wav")
        clips = [self._clip(str(path), "AAAA1111", 1, 1.0),
                 self._clip(str(path), "BBBB2222", 2, 5.0)]
        moves, ambiguous = organize.plan_prepend_ids(build_takes(clips))
        self.assertEqual(moves, [])
        self.assertEqual(ambiguous[path], ["AAAA1111", "BBBB2222"])

    def test_exports_are_well_formed(self):
        clips = [self._clip("a.mp4", "AAAA1111", 1, 5.0, has_video=True,
                            frame_rate=24.0, width=1920, height=1080),
                 self._clip("b.mp4", "AAAA1111", 1, 2.0, has_video=True,
                            frame_rate=24.0, width=1920, height=1080),
                 self._clip("c.wav", "BBBB2222", 2, 1.0)]
        takes = build_takes(clips)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            csv_path = nle.write_csv(takes, td / "o.csv")
            rows = csv_path.read_text().strip().splitlines()
            self.assertEqual(len(rows), 1 + len(clips))

            json_path = nle.write_json(takes, td / "o.json")
            import json as _json
            data = _json.loads(json_path.read_text())
            self.assertEqual(len(data["takes"]), 2)

            xml_path = nle.write_fcpxml(takes, td / "o.fcpxml")
            root = ET.fromstring(xml_path.read_text())
            self.assertEqual(root.tag, "fcpxml")
            self.assertEqual(len(root.findall(".//project")), 2)
            # Every clip reference must resolve to a declared asset.
            ids = {a.get("id") for a in root.findall(".//asset")}
            for ref in root.findall(".//asset-clip"):
                self.assertIn(ref.get("ref"), ids)

            edls = nle.write_edls(takes, td / "edl")
            self.assertEqual(len(edls), 2)
            self.assertIn("TITLE:", edls[0].read_text())

    def test_fcpxml_offsets_are_never_negative(self):
        clips = [self._clip(f"{i}.mp4", "AAAA1111", 1, float(i), has_video=True,
                            frame_rate=25.0, width=1920, height=1080)
                 for i in range(1, 5)]
        takes = build_takes(clips)
        with tempfile.TemporaryDirectory() as td:
            xml_path = nle.write_fcpxml(takes, Path(td) / "o.fcpxml")
            root = ET.fromstring(xml_path.read_text())
            for ref in root.findall(".//asset-clip"):
                value = ref.get("offset", "0s").rstrip("s")
                num = float(value.split("/")[0])
                self.assertGreaterEqual(num, 0, "negative offset in FCPXML")


if __name__ == "__main__":
    unittest.main(verbosity=2)
