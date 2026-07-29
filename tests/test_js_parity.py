"""Prove the JavaScript encoder and the Python one agree.

The webapp emits chirps that the Python parser has to read months later, so a
one-bit divergence between the two implementations would be a silent, total
failure in the field.  This renders waveforms with the JS encoder under Node,
decodes them with Python, and also compares the intermediate stages directly so
a mismatch points at the stage that broke rather than just "it did not decode".

Skips itself if Node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from chirpsync import codec, css, detect, frame  # noqa: E402

JS = ROOT / "webapp" / "chirpsync.js"

NODE_SCRIPT = r"""
const path = require('path');
const CS = require(process.argv[2]);
const spec = JSON.parse(process.argv[3]);

if (spec.mode === 'stages') {
  const data = Uint8Array.from(spec.bytes);
  console.log(JSON.stringify({
    crc16: CS.crc16(data),
    whiten: Array.from(CS.whiten(data)),
    conv: Array.from(CS.convEncode(Uint8Array.from(spec.bits))),
    block: Array.from(CS.encodeBlock(data, spec.sf))
  }));
} else {
  const out = CS.generate({
    profile: spec.profile,
    takeId: spec.takeId ? Uint8Array.from(spec.takeId) : undefined,
    sampleRate: spec.sampleRate,
    leadIn: spec.leadIn,
    leadOut: spec.leadOut
  });
  const fs = require('fs');
  fs.writeFileSync(spec.out, Buffer.from(new Float32Array(out.audio).buffer));
  console.log(JSON.stringify({
    take: out.take, duration: out.duration,
    syncOffsets: out.syncOffsets, layout: out.layout,
    samples: out.audio.length, keys: Object.keys(out)
  }));
}
"""


def node_available() -> bool:
    return shutil.which("node") is not None and JS.exists()


def run_node(spec: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "run.js"
        script.write_text(NODE_SCRIPT)
        proc = subprocess.run(
            ["node", str(script), str(JS), json.dumps(spec)],
            capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:])
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if spec.get("out"):
            result["audio"] = np.fromfile(spec["out"], dtype="<f4").astype(float)
        return result


@unittest.skipUnless(node_available(), "node or webapp/chirpsync.js not available")
class JsParityTest(unittest.TestCase):

    def test_javascript_default_profile_is_balanced(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "js.f32")
            js = run_node({
                "mode": "render", "sampleRate": 48000,
                "leadIn": 0.0, "leadOut": 0.0, "out": out,
            })
        self.assertEqual(js["layout"]["duration"], 1.984)

    def test_intermediate_stages_match(self):
        rng = np.random.default_rng(4)
        for trial in range(6):
            payload = bytes(rng.integers(0, 256, size=int(rng.integers(6, 40))))
            bits = rng.integers(0, 2, size=int(rng.integers(16, 120))).astype(np.int8)
            sf = int(rng.choice([6, 7, 8, 9]))
            got = run_node({"mode": "stages", "bytes": list(payload),
                            "bits": [int(b) for b in bits], "sf": sf})
            with self.subTest(trial=trial, sf=sf):
                self.assertEqual(got["crc16"], codec.crc16(payload), "crc16")
                self.assertEqual(bytes(got["whiten"]), __import__(
                    "chirpsync.fec", fromlist=["whiten"]).whiten(payload), "whitening")
                from chirpsync import fec
                self.assertEqual(got["conv"], [int(b) for b in fec.conv_encode(bits)],
                                 "convolutional encoder")
                self.assertEqual(
                    got["block"],
                    [int(v) for v in frame.encode_block(payload, sf)],
                    "block -> symbols")

    def test_waveform_matches_python_sample_for_sample(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "js.f32")
            take_id = 0x0123456789
            spec = {
                "mode": "render", "profile": "balanced",
                "takeId": list(take_id.to_bytes(5, "big")),
                "sampleRate": 48000,
                "leadIn": 0.0, "leadOut": 0.0, "out": out,
            }
            js = run_node(spec)

            from chirpsync.generator import generate
            py = generate(profile="balanced", take_id=take_id,
                          sample_rate=48000,
                          lead_in=0.0, lead_out=0.0)

            self.assertEqual(js["take"], py.payload.take)
            self.assertEqual(js["audio"].size, py.audio.size)
            # float32 round-trip and JS double maths give a tiny amplitude
            # difference; anything structural would be orders of magnitude worse.
            peak_err = float(np.max(np.abs(js["audio"] - py.audio)))
            self.assertLess(peak_err, 2e-4, f"waveforms diverge by {peak_err}")

    def test_python_decodes_javascript_chirps(self):
        for profile in ("fast", "balanced", "robust"):
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "js.f32")
                js = run_node({
                    "mode": "render", "profile": profile,
                    "sampleRate": 44100,
                    "leadIn": 0.3, "leadOut": 0.3, "out": out,
                })
            dets = detect(js["audio"], 44100, profiles=[profile])
            with self.subTest(profile=profile):
                self.assertEqual(len(dets), 1, "expected exactly one chirp")
                self.assertEqual(dets[0].take, js["take"])
                self.assertAlmostEqual(dets[0].time, js["syncOffsets"][0], places=3)
                self.assertNotIn("uuid", js["keys"])
                self.assertNotIn("timestamp", js["keys"])
                self.assertNotIn("metadata", js["keys"])

    def test_javascript_chirp_survives_a_room(self):
        """The realistic case: played from a phone, heard across a room."""
        import simulate as S

        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "js.f32")
            js = run_node({
                "mode": "render", "profile": "balanced",
                "sampleRate": 48000,
                "leadIn": 0.5, "leadOut": 1.0, "out": out,
            })
        cond = S.Condition("room", rt60=0.45, distance=3.0, noise="babble",
                           snr_db=8, mic="camera_internal", speaker="phone",
                           codec=("aac", "128k"))
        cap = S.simulate(js["audio"], 48000, js["syncOffsets"][0], cond,
                         burst_duration=js["layout"]["duration"])
        dets = detect(cap.audio, cap.fs, profiles=["balanced"])
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].take, js["take"])
        self.assertLess(abs(dets[0].time - cap.true_time), 0.002)

    def test_duration_estimate_matches_render(self):
        for profile in ("fast", "balanced", "robust"):
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "js.f32")
                js = run_node({
                    "mode": "render", "profile": profile,
                    "sampleRate": 48000,
                    "leadIn": 0.0, "leadOut": 0.0, "out": out,
                })
            prof = css.get_profile(profile)
            expected = (prof.preamble + prof.sfd +
                        frame.block_symbols(codec.PAYLOAD_BYTES, prof.sf)
                        ) * prof.symbol_time
            with self.subTest(profile=profile):
                self.assertAlmostEqual(js["layout"]["duration"], expected, places=6)
                self.assertAlmostEqual(js["duration"], js["layout"]["duration"],
                                       places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
