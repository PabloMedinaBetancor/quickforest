"""Pins the defaults to docs/BUILD-STANDARD.md (the QuickForest standard build): a default that drifts from the file fails here."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backend_codegen as bc

DOC = (ROOT / "docs" / "BUILD-STANDARD.md").read_text()


class BuildStandardTest(unittest.TestCase):
    def test_the_file_states_the_rules(self):
        for text in ("fat LTO + 16 codegen units (all threads) + inline threshold 3000", "ALL the data", "DISJOINT", "more than 1 %"):
            self.assertIn(text, DOC)

    def test_generated_code_default(self):
        s = bc.resolve_compile({})
        self.assertEqual((s["lto"], s["codegen_units"], s["inline_threshold"], s["pgo"]), ("fat", 16, 3000, "auto"))
        self.assertEqual(bc.DEFAULT_COMPILE_PRESET, "1.1")
        self.assertIn('DEFAULT_COMPILE_PRESET = "1.1"', DOC)
        self.assertIn("`inline_threshold 3000`", DOC)

    def test_cspgo_default(self):
        self.assertEqual(bc.resolve_compile({})["cspgo"], "auto")  # on wherever it can be built
        self.assertEqual(bc.resolve_compile({"cspgo": "off"})["cspgo"], "off")
        self.assertIn("--cspgo on|off", DOC)
        self.assertIn('"cspgo"', (ROOT / "scripts" / "backend_codegen.py").read_text())

    def test_evaluator_profile(self):
        cargo = (ROOT / "rust" / "Cargo.toml").read_text()
        profile = cargo.split("[profile.release]", 1)[1].split("\n[", 1)[0]
        self.assertRegex(profile, r'lto\s*=\s*"fat"')
        self.assertRegex(profile, r"codegen-units\s*=\s*16\b")
        flags = (ROOT / "rust" / ".cargo" / "config.toml").read_text()
        self.assertIn("-C", flags)
        self.assertIn("llvm-args=-inline-threshold=3000", flags)
        build = (ROOT / "rust" / "build.sh").read_text()
        self.assertIn("unset RUSTFLAGS", build)  # a user's flags never change the shipped library; the config file stays in force
        self.assertNotIn("inline-threshold", build.replace("unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS", ""))  # one home for the flag

    def test_benchmark_profile_rows_are_disjoint_from_the_timed_rows(self):
        src = (ROOT / "scripts" / "compare.py").read_text()
        # prepare(): the timed rows are perm[:n_speed], the profile rows come from perm[n_speed:] minus the exactness pick
        self.assertRegex(src, r"spare = np\.setdiff1d\(perm\[n_speed:\], pick\)")
        self.assertIn("x-speed.npy", src)
        self.assertIn("x-pgo.npy", src)


if __name__ == "__main__":
    unittest.main()
