"""The CS-PGO switch (QF standard, docs/BUILD-STANDARD.md rule 4): `cspgo` on | off, on by default wherever the build is profile-guided, the
fat-LTO chunks shape and the toolchain can do it; like `pgo`. Settings and refusals are checked without compiling; the real toy builds (CS-PGO on,
off) need rustc with rust-lld and are skipped without them.

    python3 -s -m unittest test_cspgo_switch
"""

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as bc
from test_compile_settings import FIX, FakeRustc, forest
from tiled import tiled


class SettingsTest(unittest.TestCase):
    def test_on_by_default_in_both_presets_and_switchable_like_pgo(self):
        self.assertEqual(bc.resolve_compile({})["cspgo"], "auto")
        self.assertEqual(bc.resolve_compile({"compile_preset": "1.0"})["cspgo"], "auto")
        for given, want in (("on", "on"), ("off", "off"), (True, "on"), (False, "off"), (None, "auto"), ("", "auto")):
            self.assertEqual(bc.resolve_compile({"cspgo": given})["cspgo"], want, given)
        with self.assertRaisesRegex(ValueError, r"--cspgo 'maybe': give on"):
            bc.resolve_compile({"cspgo": "maybe"})

    def test_the_switch_is_a_compile_key_and_a_command_line_flag(self):
        import argparse

        self.assertIn("cspgo", bc.COMPILE_KEYS)
        ap = argparse.ArgumentParser()
        bc.add_compile_arguments(ap)
        self.assertEqual(bc.compile_options(ap.parse_args(["--cspgo", "off"])), {"cspgo": "off"})
        self.assertEqual(bc.compile_options(ap.parse_args([])), {})

    def test_the_switch_does_not_change_the_build_shape(self):
        self.assertEqual(bc.resolve_compile({"cspgo": "off"})["preset"], "1.1")

    def test_cspgo_on_is_refused_where_it_cannot_apply(self):
        f = forest()
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-cs-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for backend, opts, text in (
            (bc.CodegenBlockBackend, {"cspgo": "on"}, "second profile pass"),
            (bc.CodegenPgoBackend, {"cspgo": "on", "pgo": "off"}, "contradict|second profile pass"),
            (bc.CodegenPgoBackend, {"cspgo": "on", "unit": "single"}, "chunks build"),
        ):
            with self.assertRaisesRegex(ValueError, text):
                backend().compile(f, tmp / "x.so", {"pgo_rows": str(tmp / "r.npy"), **opts})


class PlanTest(unittest.TestCase):
    """With the toolchain faked: CS-PGO runs by default, `off` is plain PGO, a shape it cannot have falls back with a notice, `on` there is refused."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-cs-"))
        cls.f = forest()
        cls.rows = cls.tmp / "rows.npy"
        np.save(cls.rows, tiled(np.load(FIX / "x.npy")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, available=True, **opts):
        FakeRustc(self)
        calls = []

        def fake_stages(build, rustc, src, merged, parts):
            calls.append(list(parts))
            so = build([], "csgen", None, lpl=True, link_extra=["-C", "link-arg=cs"], strip=True)
            return so, bc.Watch(), {"cs_profile_functions": 7, "cs_seconds": {}}

        patches = [
            mock.patch.object(bc, "cspgo_unavailable", lambda *a: None if available else "no lld here"),
            mock.patch.object(bc, "cspgo_stages", fake_stages),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with contextlib.redirect_stderr(io.StringIO()):
            info = bc.CodegenPgoBackend().compile(
                self.f, self.tmp / "m.so", {"jobs": 1, "ram_mb": 100_000, "pgo_rows": str(self.rows), **opts}
            )
        return calls, info

    def test_cspgo_runs_by_default(self):
        calls, info = self.build()
        self.assertEqual(len(calls), 1)
        self.assertTrue(info["cspgo"])
        self.assertEqual(info["cs_profile_functions"], 7)
        self.assertIn("CS-PGO", info["note"])

    def test_off_is_plain_pgo(self):
        calls, info = self.build(cspgo="off")
        self.assertEqual(calls, [])
        self.assertFalse(info["cspgo"])
        self.assertTrue(info["pgo"])
        self.assertNotIn("cspgo_note", info)

    def test_a_toolchain_without_it_falls_back_to_plain_pgo_with_a_notice(self):
        calls, info = self.build(available=False)
        self.assertEqual(calls, [])
        self.assertFalse(info["cspgo"])
        self.assertIn("no lld here", info["cspgo_note"])
        with self.assertRaisesRegex(ValueError, "--cspgo on: no lld here"):
            self.build(available=False, cspgo="on")


class UnavailableTest(unittest.TestCase):
    def test_reasons(self):
        s = bc.resolve_compile({})
        self.assertIn("chunks build", bc.cspgo_unavailable(s, "single", "rustc"))
        self.assertIn("fat-LTO", bc.cspgo_unavailable(bc.resolve_compile({"compile_preset": "1.0"}), "chunks", "rustc"))


@unittest.skipUnless(shutil.which("rustc") or __import__("os").environ.get("RUST_DIR"), "needs rustc")
class RealBuildTest(unittest.TestCase):
    """Both paths on the real toolchain (a few minutes at one job): CS-PGO on and off both build, load and answer bit-for-bit alike."""

    def test_both_paths_build_and_agree(self):
        try:
            rustc = str(bc.toolchain_bin() / "rustc")
            bc.llvm_profdata()
        except (bc.ToolchainMissing, Exception) as e:
            self.skipTest(str(e))
        why = bc.cspgo_unavailable(bc.resolve_compile({}), "chunks", rustc)
        if why:
            self.skipTest(why)
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-cs-real-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        f, x = forest(), np.load(FIX / "x.npy")
        rows = tmp / "rows.npy"
        np.save(rows, tiled(x))
        out, infos = {}, {}
        for tag in ("on", "off"):
            with contextlib.redirect_stderr(io.StringIO()):
                infos[tag] = bc.CodegenPgoBackend().compile(f, tmp / f"{tag}.so", {"jobs": 1, "pgo_rows": str(rows), "cspgo": tag})
            out[tag] = bc.CodegenForest(tmp / f"{tag}.so").predict(x, 1)
        self.assertTrue(infos["on"]["cspgo"])
        self.assertGreater(infos["on"]["cs_profile_functions"], 0)
        self.assertFalse(infos["off"]["cspgo"])
        self.assertTrue(np.array_equal(out["on"].view(np.uint32), out["off"].view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
