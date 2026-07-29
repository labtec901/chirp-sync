"""Command-line behavior and release-facing defaults."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chirpsync import codec, css, generate, media  # noqa: E402
from chirpsync.cli import build_parser, main  # noqa: E402


class CliTest(unittest.TestCase):

    def test_gen_defaults_to_fast(self):
        args = build_parser().parse_args(["gen"])
        self.assertEqual(css.DEFAULT_PROFILE, "fast")
        self.assertEqual(args.profile, "fast")

    def test_prepend_id_dry_run_then_apply(self):
        take = "X4THAPJ9"
        chirp = generate(
            profile="fast",
            take_id=codec.take_id_from_str(take),
            lead_in=0.15,
            lead_out=0.15,
        )
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "camera.wav"
            media.write_wav(source, chirp.audio, chirp.sample_rate, "pcm16")
            target = source.with_name(f"{take}_{source.name}")

            self.assertEqual(main(["scan", str(source), "--profile", "fast",
                                   "--prepend-id", "--dry-run"]), 0)
            self.assertTrue(source.exists())
            self.assertFalse(target.exists())

            self.assertEqual(main(["scan", str(source), "--profile", "fast",
                                   "--prepend-id"]), 0)
            self.assertFalse(source.exists())
            self.assertTrue(target.exists())

    def test_website_and_cli_contain_no_em_dash(self):
        paths = [
            ROOT / "src" / "chirpsync" / "cli.py",
            ROOT / "webapp" / "index.html",
            ROOT / "webapp" / "chirpsync.js",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("\N{EM DASH}", text)
                self.assertNotIn("&mdash;", text.lower())

    def test_website_requests_mobile_playback_audio_session(self):
        page = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
        self.assertIn("navigator.audioSession.type = 'playback'", page)
        self.assertIn("turn off Silent Mode and raise media volume", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
