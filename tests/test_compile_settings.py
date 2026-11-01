"""The compile settings of QF 1.1 (`lto`, `codegen_units`, `inline_threshold`, `pgo`, the `1.0` / `1.1` presets) and the new default build.

Python only and light: the flag assembly, the refusals, the build info and the RAM plan are checked without compiling anything (rustc is replaced by
a recorder that writes the files the build expects). The one test that needs the real rustc builds a toy forest at one job and is skipped without it.

    python3 -s -m unittest test_compile_settings
"""

import contextlib
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import argparse

import backend_codegen as bc
import compile as qf_compile
import optimize
import qf_common as qf
import qf_formats
from tiled import tiled

FIX = HERE / "fixtures"
MODEL = FIX / "lgb_plain.txt"  # a LightGBM text model: it loads without xgboost


def forest():
    return qf.load_forest(qf_formats.model_files(MODEL), "output", False)


class FakeRustc:
    """Stands in for rustc and the profile tools: records every command, writes the object / library the build looks for, and lets the profile
    gate and the profile run pass. What the real tools do is checked by builds with the real rustc (test_codegen_target.py), not here."""

    def __init__(self, test):
        self.cmds: list[list[str]] = []
        for name, fake in {
            "toolchain_bin": lambda: Path("/fake/bin"),
            "target_features": lambda rustc, cpu: [],
            "missing_features": lambda feats: [],
            "llvm_profdata": lambda: Path("/fake/llvm-profdata"),
            "run_measured": self.run_measured,
            "profile_run": lambda inst, rows, raw, kind: raw.mkdir(),
            "merge_profiles": lambda merged, raw: None,
            "edge_overlap": lambda a, b: 99.0,
            "function_overlap": lambda a, b: 99.0,
            "profile_table": lambda p: {"c0::b1": ("h", 5)},
            "check_profile": lambda watch, merged, plan, mode: {"generated": 1, "hot": 1, "cold": 0},
        }.items():
            p = mock.patch.object(bc, name, fake)
            p.start()
            test.addCleanup(p.stop)

    def run_measured(self, cmd, env=None, watch=None):
        self.cmds.append(list(cmd))
        if "--out-dir" in cmd:
            (Path(cmd[cmd.index("--out-dir") + 1]) / "c.o").write_bytes(b"obj")
        if "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"lib")
        return 0, "", 123.0

    def flags(self, kind, phase):
        """The `-C` values of every chunk build (kind "chunk") or driver link ("link") of one phase: profile-generate, profile-use, or none."""
        want = {"gen": "profile-generate=", "use": "profile-use=", "plain": None}[phase]
        out = []
        for c in self.cmds:
            if ("cdylib" in c) != (kind == "link"):
                continue
            has = [a for a in c if a.startswith(("profile-generate=", "profile-use="))]
            if (want is None and not has) or (want is not None and has and has[0].startswith(want)):
                out.append(c)
        return out


def settings_flags(cmd):
    """What the shape of a build is, from a rustc command line: the opt-level, LTO, codegen-unit and inline flags, in order."""
    return [
        a
        for i, a in enumerate(cmd)
        if i and cmd[i - 1] == "-C" and a.startswith(("opt-level=", "lto=", "codegen-units=", "llvm-args=-inline"))
    ]


class FlagAssemblyTest(unittest.TestCase):
    def test_the_default_is_the_1_1_preset_behind_one_constant(self):
        self.assertEqual(bc.DEFAULT_COMPILE_PRESET, "1.1")
        self.assertIs(bc.DEFAULT_COMPILE, bc.COMPILE_PRESETS[bc.DEFAULT_COMPILE_PRESET])
        self.assertEqual(
            {k: bc.DEFAULT_COMPILE[k] for k in ("lto", "codegen_units", "inline_threshold", "pgo")},
            {"lto": "fat", "codegen_units": 16, "inline_threshold": 3000, "pgo": "auto"},
        )
        s = bc.resolve_compile({})
        self.assertEqual((s["preset"], s["lto"], s["codegen_units"], s["inline_threshold"]), ("1.1", "fat", 16, 3000))
        # fat LTO is a no-op on a chunk crate: the chunks build (the default) does not pass it; the driver's link and the one-crate build do
        self.assertEqual(
            bc.chunk_flags(3, s, 16), ["-C", "opt-level=3", "-C", "codegen-units=16", "-C", "llvm-args=-inline-threshold=3000"]
        )
        one = bc.resolve_compile({"unit": "single"})
        self.assertEqual(
            bc.chunk_flags(3, one, 16),
            ["-C", "opt-level=3", "-C", "lto=fat", "-C", "codegen-units=16", "-C", "llvm-args=-inline-threshold=3000"],
        )
        self.assertEqual(
            bc.driver_flags(s), ["-C", "lto=fat", "-C", "llvm-args=-inline-threshold=3000"]
        )  # rustc's own 16 units, no codegen-units=1
        by_hand = bc.resolve_compile({"codegen_units": 1})
        self.assertEqual(
            bc.driver_flags(by_hand), ["-C", "codegen-units=1", "-C", "lto=fat", "-C", "llvm-args=-inline-threshold=3000"]
        )  # a count the user gave is the driver's too
        four = bc.resolve_compile({"codegen_units": 4})
        self.assertEqual(bc.driver_flags(four)[:2], ["-C", "codegen-units=4"])

    def test_flipping_the_one_constant_gives_QF_1_0s_build_with_exactly_its_flags(self):
        with mock.patch.object(bc, "DEFAULT_COMPILE_PRESET", "1.0"):
            s = bc.resolve_compile({})
        self.assertEqual((s["preset"], s["shape"]), ("1.0", "fast"))
        self.assertEqual(
            bc.chunk_flags(3, s, 16), ["-C", "opt-level=3", "-C", "lto=off", "-C", "codegen-units=16"]
        )  # what 1.0 passed to every chunk
        self.assertEqual(bc.driver_flags(s), [])  # and nothing extra to the driver's link
        self.assertEqual(bc.resolve_compile({"compile_preset": "1.0"})["shape"], "fast")

    def test_the_fast_shape_is_QF_1_0s_build_and_top_is_removed(self):
        self.assertEqual(bc.SHAPES, {"fast": 16})
        fast = bc.resolve_compile({"shape": "fast"})
        self.assertEqual((fast["preset"], fast["codegen_units"], fast["lto"], fast["inline_threshold"]), ("1.0", 16, "off", None))
        with self.assertRaisesRegex(ValueError, r"--shape top: removed .*every build uses 16 codegen units.*--shape fast"):
            bc.resolve_compile({"shape": "top"})
        by_hand = bc.resolve_compile({"lto": "off", "codegen_units": 1, "inline_threshold": "default"})  # one unit, only when asked
        self.assertEqual(bc.chunk_flags(3, by_hand, 1), ["-C", "opt-level=3", "-C", "lto=off", "-C", "codegen-units=1"])
        self.assertEqual(bc.driver_flags(by_hand), ["-C", "codegen-units=1"])
        # no preset, shape or default builds with one codegen unit
        self.assertTrue(all(p["codegen_units"] > 1 for p in bc.COMPILE_PRESETS.values()))
        self.assertTrue(all(p["codegen_units"] > 1 for p in bc.SHAPE_BUILDS.values()))

    def test_every_combination_of_the_settings_gives_its_exact_flags(self):
        for lto, units, inline in itertools.product(("off", "fat"), (1, 4, 16), (None, 800, 3000)):
            s = bc.resolve_compile({"lto": lto, "codegen_units": units, "inline_threshold": "default" if inline is None else inline})
            tail = ["-C", f"codegen-units={units}"] + (["-C", f"llvm-args=-inline-threshold={inline}"] if inline else [])
            chunk = [
                "-C",
                "opt-level=3",
                *(["-C", "lto=off"] if lto == "off" else []),
                *tail,
            ]  # chunks: fat LTO is a no-op there, so it is not passed
            self.assertEqual(bc.chunk_flags(3, s, units), chunk, (lto, units, inline))
            one = bc.resolve_compile(
                {"unit": "single", "lto": lto, "codegen_units": units, "inline_threshold": "default" if inline is None else inline}
            )
            self.assertEqual(
                bc.chunk_flags(3, one, units), ["-C", "opt-level=3", "-C", f"lto={lto}", *tail], (lto, units, inline, "single")
            )
            link = (["-C", f"codegen-units={units}"] if units != 16 else []) + (["-C", "lto=fat"] if lto == "fat" else [])
            link += ["-C", f"llvm-args=-inline-threshold={inline}"] if inline else []
            self.assertEqual(bc.driver_flags(s), link, (lto, units, inline))

    def test_a_single_option_overrides_the_preset_and_a_shape_replaces_the_whole_build(self):
        self.assertEqual(bc.resolve_compile({"lto": "off"})["inline_threshold"], 3000)  # the rest of the preset stays
        self.assertEqual(bc.resolve_compile({"inline_threshold": "default"})["inline_threshold"], None)
        self.assertEqual(
            bc.resolve_compile({"inline_threshold": "0"})["inline_threshold"], 0
        )  # 0 is a threshold; None / "default" is rustc's own
        self.assertEqual(bc.inline_flags(bc.resolve_compile({"inline_threshold": 0})), ["-C", "llvm-args=-inline-threshold=0"])
        shape_then_one = bc.resolve_compile({"shape": "fast", "codegen_units": 8})
        self.assertEqual((shape_then_one["lto"], shape_then_one["codegen_units"], shape_then_one["shape"]), ("off", 8, "custom"))
        self.assertEqual(
            bc.resolve_compile({"shape": "fast", "compile_preset": "1.1"})["lto"], "off"
        )  # the shape replaces the preset's build
        self.assertEqual(bc.resolve_compile({"pgo": "off"})["pgo"], "off")
        self.assertEqual(bc.resolve_compile({"pgo": True})["pgo"], "on")
        self.assertEqual(bc.resolve_compile({})["pgo"], "auto")

    def test_the_description_names_the_build_in_the_notes(self):
        self.assertEqual(bc.describe_compile(bc.COMPILE_PRESETS["1.1"]), "fat LTO, 16 codegen units, inline 3000")
        self.assertEqual(bc.describe_compile(bc.COMPILE_PRESETS["1.0"]), "no LTO, 16 codegen units, rustc's inlining")

    def test_a_bad_setting_is_refused_in_one_line(self):
        for opts, text in (
            ({"lto": "thin"}, "--lto 'thin': give off or fat"),
            ({"codegen_units": 0}, "--codegen-units 0"),
            ({"codegen_units": "many"}, "--codegen-units 'many'"),
            ({"codegen_units": 1.5}, "--codegen-units 1.5"),
            ({"codegen_units": 10_000}, "--codegen-units 10000"),
            ({"inline_threshold": -1}, "--inline-threshold -1"),
            ({"inline_threshold": "lots"}, "--inline-threshold 'lots'"),
            ({"inline_threshold": 10**9}, "--inline-threshold 1000000000"),
            ({"compile_preset": "2.0"}, "--compile-preset '2.0': give one of 1.1, 1.0"),
            ({"pgo": "maybe"}, "--pgo 'maybe': give on"),
            ({"shape": "turbo"}, "--shape 'turbo'"),
        ):
            with self.assertRaisesRegex(ValueError, text.replace("(", r"\(")):
                bc.resolve_compile(opts)


class SharedShapeTest(unittest.TestCase):
    """The instrumented build gets exactly the final's shape: the same lto, units and inline flags in every chunk build and in the driver's link,
    with `profile-generate` in place of `profile-use`; the plain build has the same shape too."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-settings-"))
        cls.f = forest()
        cls.rows = cls.tmp / "rows.npy"
        np.save(cls.rows, tiled(np.load(FIX / "x.npy")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, backend, **opts):
        fake = FakeRustc(self)
        with contextlib.redirect_stderr(io.StringIO()):
            info = backend().compile(self.f, self.tmp / "m.so", {"jobs": 1, "ram_mb": 100_000, **opts})
        return fake, info

    def test_the_instrumented_and_the_final_build_share_the_shape_in_every_combination(self):
        for lto, units, inline in itertools.product(("off", "fat"), (1, 16), (None, 3000)):
            opts = {"lto": lto, "codegen_units": units, "inline_threshold": "default" if inline is None else inline}
            fake, info = self.build(bc.CodegenPgoBackend, pgo_rows=str(self.rows), **opts)
            s = bc.resolve_compile(opts)
            for kind in ("chunk", "link"):
                gen, use = fake.flags(kind, "gen"), fake.flags(kind, "use")
                self.assertTrue(gen and use, (opts, kind))
                shapes = {tuple(settings_flags(c)) for c in gen + use}
                self.assertEqual(len(shapes), 1, (opts, kind, shapes))
                if kind == "chunk":
                    self.assertEqual(list(shapes.pop()), bc.chunk_flags(3, s, units)[1::2], opts)
                else:
                    want = bc.driver_flags(s)
                    self.assertEqual([a for a in list(shapes.pop()) if not a.startswith("opt-level")], want[1::2], opts)
            # the profile-generate build carries the generate flag and nothing of the final, the final carries profile-use and the gate's warning flag
            self.assertTrue(all(not any(a.startswith("profile-use=") for a in c) for c in fake.flags("chunk", "gen")))
            self.assertTrue(all(any("pgo-warn-missing-function" in a for a in c) for c in fake.flags("chunk", "use")))
            self.assertEqual((info["lto"], info["codegen_units"], info["inline_threshold"], info["pgo"]), (lto, units, inline, True))

    def test_a_build_without_a_profile_has_the_same_shape_as_the_profile_guided_final(self):
        plain, _ = self.build(bc.CodegenBlockBackend)
        guided, _ = self.build(bc.CodegenPgoBackend, pgo_rows=str(self.rows))
        a = {tuple(settings_flags(c)) for c in plain.flags("chunk", "plain")}
        b = {tuple(settings_flags(c)) for c in guided.flags("chunk", "use")}
        self.assertEqual(a, b)
        self.assertNotIn("lto=fat", a.pop())  # chunks build: fat LTO acts on the driver's link only (test below)

    def test_the_default_build_passes_the_new_default_flags_to_rustc(self):
        fake, _ = self.build(bc.CodegenBlockBackend)
        (chunk,) = fake.flags("chunk", "plain")
        (link,) = fake.flags("link", "plain")
        self.assertNotIn("lto=fat", chunk)  # a no-op on a chunk crate
        self.assertIn("lto=fat", link)
        self.assertIn("codegen-units=16", chunk)
        self.assertNotIn("codegen-units=1", link)  # the driver's link keeps rustc's own 16 units; one unit is never a default
        for cmd in (chunk, link):
            self.assertIn("llvm-args=-inline-threshold=3000", cmd)
        fake10, _ = self.build(bc.CodegenBlockBackend, compile_preset="1.0")
        (chunk10,) = fake10.flags("chunk", "plain")
        (link10,) = fake10.flags("link", "plain")
        self.assertEqual(settings_flags(chunk10), ["opt-level=3", "lto=off", "codegen-units=16"])
        self.assertEqual([a for a in link10 if a.startswith(("lto=", "codegen-units=", "llvm-args"))], [])

    def test_the_build_info_records_the_settings(self):
        _, info = self.build(bc.CodegenPgoBackend, pgo_rows=str(self.rows))
        self.assertEqual(
            {k: info[k] for k in ("compile_preset", "lto", "codegen_units", "inline_threshold", "pgo", "shape")},
            {"compile_preset": "1.1", "lto": "fat", "codegen_units": 16, "inline_threshold": 3000, "pgo": True, "shape": "custom"},
        )
        self.assertEqual(info["build"], "fat LTO, 16 codegen units, inline 3000")
        self.assertGreater(info["peak_factor"], 1)
        self.assertIn("build fat LTO, 16 codegen units, inline 3000 (preset 1.1)", info["note"])
        _, old = self.build(bc.CodegenBlockBackend, compile_preset="1.0")
        self.assertEqual(
            (
                old["compile_preset"],
                old["shape"],
                old["lto"],
                old["codegen_units"],
                old["inline_threshold"],
                old["pgo"],
                old["peak_factor"],
            ),
            ("1.0", "fast", "off", 16, None, False, 1.0),
        )
        _, odd = self.build(bc.CodegenBlockBackend, lto="off", codegen_units=2, inline_threshold=500)
        self.assertEqual((odd["compile_preset"], odd["shape"]), ("custom", "custom"))
        self.assertNotIn("preset", odd["note"].split(", block")[0])

    def test_a_failed_instrumented_build_says_it_does_not_fall_back_to_another_shape(self):
        fake = FakeRustc(self)

        def failing(cmd, env=None, watch=None):
            fake.cmds.append(cmd)
            return (
                (1, "fat LTO is not possible here", 1.0)
                if any(a.startswith("profile-generate=") for a in cmd)
                else fake.run_measured(cmd, env, watch)
            )

        with mock.patch.object(bc, "run_measured", failing), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(bc.BuildFailed, "does not fall back to another shape.*--compile-preset 1.0"):
                bc.CodegenPgoBackend().compile(self.f, self.tmp / "no.so", {"pgo_rows": str(self.rows), "ram_mb": 100_000})
        self.assertFalse((self.tmp / "no.so").exists())
        self.assertEqual(len([c for c in fake.cmds if "profile-use" in " ".join(c)]), 0, "no final build was tried on another shape")

    def test_pgo_on_and_off_must_agree_with_the_backend(self):
        with self.assertRaisesRegex(ValueError, "--pgo on asks for a profile-guided build and codegen-block is not one"):
            bc.CodegenBlockBackend().compile(self.f, self.tmp / "x.so", {"pgo": "on"})
        with self.assertRaisesRegex(ValueError, "--pgo off and --backend codegen-pgo contradict"):
            bc.CodegenPgoBackend().compile(self.f, self.tmp / "x.so", {"pgo": "off", "pgo_rows": str(self.rows)})
        _, info = self.build(bc.CodegenPgoBackend, pgo="on", pgo_rows=str(self.rows))
        self.assertTrue(info["pgo"])


class RefusalsTest(unittest.TestCase):
    def test_the_flat_and_wave_backends_refuse_every_setting(self):
        for backend in ("flat", "wave", "wave-avx2"):
            for opt in (
                {"lto": "fat"},
                {"codegen_units": 1},
                {"inline_threshold": 3000},
                {"pgo": "on"},
                {"compile_preset": "1.0"},
                {"shape": "fast"},
            ):
                with self.subTest(backend=backend, option=opt):
                    with self.assertRaisesRegex(ValueError, "how generated code is built: .* is not generated code"):
                        qf_compile.compile_model(MODEL, Path(tempfile.gettempdir()) / "quickforest-never.bin", backend, **opt)

    def test_the_refusal_names_the_flags_that_were_given(self):
        with self.assertRaisesRegex(
            ValueError, r"--lto, --inline-threshold choose how generated code is built: flat is not generated code"
        ):
            qf_compile.compile_model(MODEL, Path(tempfile.gettempdir()) / "quickforest-never.bin", "flat", lto="fat", inline_threshold=3000)

    def test_a_mistyped_setting_is_found_before_the_model_is_read(self):
        for opt in ({"lto": "thin"}, {"codegen_units": 0}, {"inline_threshold": "x"}, {"compile_preset": "9"}, {"pgo": "x"}):
            with self.assertRaises(ValueError):
                qf_compile.compile_model(MODEL, Path(tempfile.gettempdir()) / "quickforest-never.bin", "codegen-block", **opt)
            with self.assertRaises(ValueError):
                optimize.optimize(MODEL, "unused.npz", Path(tempfile.gettempdir()) / "quickforest-never-out", "100%", True, **opt)

    def test_settings_without_compile_are_refused_by_optimize(self):
        with self.assertRaisesRegex(ValueError, "--lto shapes the compiled forest: add --compile"):
            optimize.optimize(MODEL, "unused.npz", Path(tempfile.gettempdir()) / "quickforest-never-out", "100%", False, lto="fat")

    def test_backend_auto_refuses_pgo_and_takes_the_other_settings(self):
        with self.assertRaisesRegex(ValueError, "--backend auto .* no --pgo"):
            qf_compile.compile_model(MODEL, Path(tempfile.gettempdir()) / "quickforest-never.bin", "auto", data="d.npz", pgo="off")
        seen = {}

        def fake_choose(model, out, data, **kw):
            seen.update(kw["compile_options"])
            return {}

        with mock.patch.dict(sys.modules, {"choose": mock.Mock(choose=fake_choose)}):
            qf_compile.compile_model(
                MODEL, Path(tempfile.gettempdir()) / "quickforest-never.bin", "auto", data="d.npz", lto="fat", inline_threshold=800
            )
        self.assertEqual({k: seen[k] for k in ("lto", "inline_threshold")}, {"lto": "fat", "inline_threshold": 800})
        self.assertNotIn("pgo", seen)


class BestBackendTest(unittest.TestCase):
    """`best` picks the profile-guided build with rows, unless `--pgo off`; `--pgo on` is never quietly turned into something else."""

    def setUp(self):
        self.f = forest()
        for name, fake in {
            "toolchain_bin": lambda: Path("/fake/bin"),
            "llvm_profdata": lambda: Path("/fake/llvm-profdata"),
            "available_ram_mb": lambda options: options.get("ram_mb"),
        }.items():
            p = mock.patch.object(bc, name, fake)
            p.start()
            self.addCleanup(p.stop)

    def test_the_default_with_rows_is_pgo_and_without_rows_codegen_block(self):
        self.assertEqual(qf_compile.best_backend(self.f, True), ("codegen-pgo", None))
        self.assertEqual(qf_compile.best_backend(self.f, False)[0], "codegen-block")

    def test_pgo_off_is_codegen_block_with_rows_and_without_a_notice(self):
        self.assertEqual(qf_compile.best_backend(self.f, True, {"pgo": "off"}), ("codegen-block", None))

    def test_pgo_on_needs_rows_and_a_profile_tool_and_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "--pgo on runs a profile pass over real rows: give --data"):
            qf_compile.best_backend(self.f, False, {"pgo": "on"})
        gone = bc.ToolchainMissing("llvm-profdata not found")
        with mock.patch.object(bc, "llvm_profdata", side_effect=gone):
            self.assertEqual(qf_compile.best_backend(self.f, True)[0], "codegen-block")  # the default only says so in the notice
            with self.assertRaises(bc.ToolchainMissing):
                qf_compile.best_backend(self.f, True, {"pgo": "on"})

    def test_too_little_ram_for_the_heavier_default_is_said_with_the_way_out(self):
        need = max(bc.RUSTC_LINK_MB, bc.rustc_peak_estimate_mb(bc.MIN_CHUNK_NODES, 1, 1.0)) / bc.RAM_SHARE
        ram = need * 1.1  # QF 1.0's build fits, the 1.1 default (about 2.9x) does not
        self.assertEqual(qf_compile.best_backend(self.f, True, {"ram_mb": ram, "compile_preset": "1.0"})[0], "codegen-pgo")
        backend, notice = qf_compile.best_backend(self.f, True, {"ram_mb": ram})
        self.assertEqual(backend, "flat")
        self.assertIn(
            "fat LTO, 16 codegen units, inline 3000 needs more than QF 1.0's build; --compile-preset 1.0 or --lto off builds lighter",
            notice,
        )


class RamPlanTest(unittest.TestCase):
    def test_the_peak_factor_grows_with_fat_lto_and_the_inline_threshold(self):
        f = lambda **o: bc.peak_factor(bc.resolve_compile(o))  # noqa: E731
        self.assertEqual(f(compile_preset="1.0"), 1.0)
        self.assertEqual(f(shape="fast"), 1.0)
        # a chunk crate runs no LTO: the chunks build is charged for the inline threshold only; the one crate and the driver's link for fat LTO too
        self.assertEqual(f(lto="fat", inline_threshold="default"), 1.0)
        self.assertEqual(f(lto="fat", inline_threshold="default", unit="single"), bc.LTO_PEAK_FACTOR)
        self.assertEqual(bc.link_peak_factor(bc.resolve_compile({"inline_threshold": "default"})), bc.LTO_PEAK_FACTOR)
        self.assertGreater(f(lto="off", inline_threshold=3000), 1.0)
        self.assertGreater(f(lto="off", inline_threshold=3000), f(lto="off", inline_threshold=800))
        self.assertEqual(f(), f(lto="off"))  # chunks: fat LTO does not change a chunk job
        self.assertGreater(f(unit="single"), f(lto="fat", inline_threshold=800, unit="single"))
        self.assertGreater(f(unit="single"), 2.0)
        self.assertEqual(bc.link_peak_factor(bc.resolve_compile({})), f(unit="single"))
        self.assertLess(f(), bc.link_peak_factor(bc.resolve_compile({})))

    def test_the_link_of_a_chunks_build_is_planned_with_the_lto_factor_the_chunk_jobs_without(self):
        link = bc.RUSTC_LINK_MB * 2.9
        self.assertIsNone(bc.plan_build([1], 1, 1, link / bc.RAM_SHARE - 1, 1.45, 2.9))
        self.assertIsNotNone(bc.plan_build([1], 1, 1, link / bc.RAM_SHARE + 1, 1.45, 2.9))
        self.assertIsNotNone(bc.plan_build([1], 1, 1, bc.RUSTC_LINK_MB * 1.45 / bc.RAM_SHARE + 1, 1.45))

    def test_the_estimate_and_the_plan_scale_with_the_factor(self):
        est = bc.rustc_peak_estimate_mb
        self.assertEqual(est(50_000, 1, 2.5), 2.5 * est(50_000, 1))
        chunks = [50_000] * 4
        ram = 3 * est(50_000, 1) / bc.RAM_SHARE + 1  # three jobs of QF 1.0's build fit
        self.assertEqual(bc.plan_build(chunks, 3, 1, ram), (3, 1))
        self.assertEqual(bc.plan_build(chunks, 3, 1, ram, 2.9), (1, 1), "the same RAM takes fewer jobs of the heavier build")
        self.assertEqual(bc.plan_build(chunks, 3, 1, 3 * est(50_000, 1, 2.9) / bc.RAM_SHARE + 1, 2.9), (3, 1))

    def test_one_codegen_unit_cannot_be_lowered_so_only_smaller_chunks_fit_and_the_link_is_scaled_too(self):
        est = bc.rustc_peak_estimate_mb
        self.assertIsNone(bc.plan_build([150_000], 1, 1, est(150_000, 1, 2.9) / bc.RAM_SHARE - 1, 2.9))
        self.assertIsNotNone(bc.plan_build([75_000], 1, 1, est(150_000, 1, 2.9) / bc.RAM_SHARE - 1, 2.9))
        # the final link is scaled too
        self.assertIsNone(bc.plan_build([1], 1, 1, bc.RUSTC_LINK_MB * 2.9 / bc.RAM_SHARE - 1, 2.9))
        self.assertIsNotNone(bc.plan_build([1], 1, 1, bc.RUSTC_LINK_MB * 2.9 / bc.RAM_SHARE + 1, 2.9))

    def test_a_build_that_cannot_fit_says_how_to_build_lighter(self):
        fake = FakeRustc(self)
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-ram-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        with self.assertRaisesRegex(
            bc.NotEnoughRam, r"fat LTO, 16 codegen units, inline 3000\) needs about .* MB.*--compile-preset 1.0 is QF 1.0's build"
        ):
            bc.CodegenBlockBackend().compile(forest(), tmp / "m.so", {"ram_mb": 20})
        with self.assertRaises(bc.NotEnoughRam) as e:
            bc.CodegenBlockBackend().compile(forest(), tmp / "m.so", {"ram_mb": 20, "compile_preset": "1.0"})
        self.assertNotIn("--compile-preset", str(e.exception), "QF 1.0's build has nothing lighter to suggest")
        self.assertEqual(fake.cmds, [], "nothing was built")

    def test_the_ram_line_is_unchanged(self):
        fake = FakeRustc(self)
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-ram-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            bc.CodegenBlockBackend().compile(forest(), tmp / "m.so", {"ram_mb": 100_000, "jobs": 1})
        self.assertEqual(err.getvalue().strip(), "RAM: building with 1 job / 16 units")
        self.assertTrue(fake.cmds)


class CommandLineTest(unittest.TestCase):
    def parse(self, *argv):
        ap = argparse.ArgumentParser()
        bc.add_compile_arguments(ap)
        return bc.compile_options(ap.parse_args(argv))

    def test_nothing_given_is_nothing_passed(self):
        self.assertEqual(self.parse(), {})

    def test_every_flag_reaches_the_options_under_the_api_name(self):
        got = self.parse(
            "--compile-preset",
            "1.0",
            "--lto",
            "fat",
            "--codegen-units",
            "4",
            "--inline-threshold",
            "default",
            "--pgo",
            "off",
            "--shape",
            "fast",
        )
        self.assertEqual(
            got,
            {"compile_preset": "1.0", "lto": "fat", "codegen_units": 4, "inline_threshold": "default", "pgo": "off", "shape": "fast"},
        )
        s = bc.resolve_compile(got)
        self.assertEqual((s["lto"], s["codegen_units"], s["inline_threshold"], s["pgo"]), ("fat", 4, None, "off"))

    def test_the_parser_refuses_what_the_api_refuses(self):
        for argv in (("--lto", "thin"), ("--pgo", "maybe"), ("--compile-preset", "2.0"), ("--shape", "turbo"), ("--codegen-units", "x")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                self.parse(*argv)
            self.assertEqual(e.exception.code, 2, argv)

    def test_compile_py_and_optimize_py_both_have_the_flags(self):
        for script in ("compile.py", "optimize.py"):
            text = (HERE.parent / "scripts" / script).read_text()
            self.assertIn("backend_codegen.add_compile_arguments(ap)", text, script)
        text = (HERE.parent / "scripts" / "compile.py").read_text()
        self.assertIn("**backend_codegen.compile_options(a)", text)

    def test_compile_py_exits_3_on_a_setting_the_backend_cannot_take(self):
        with mock.patch.object(
            sys,
            "argv",
            ["compile.py", str(MODEL), str(Path(tempfile.gettempdir()) / "quickforest-never.bin"), "--backend", "flat", "--lto", "fat"],
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
                qf_compile.main()
        self.assertEqual(e.exception.code, 3)
        self.assertIn("--lto chooses how generated code is built: flat is not generated code", err.getvalue())


class ReviewR1Test(unittest.TestCase):
    """Regression tests for: `--opt lto=off`, `--pgo on` never turned into something else, the final build's way out, the numpy folders."""

    setUpClass = classmethod(SharedShapeTest.setUpClass.__func__)
    tearDownClass = classmethod(SharedShapeTest.tearDownClass.__func__)
    build = SharedShapeTest.build

    def test_lto_given_as_a_boolean_by_the_opt_parser_is_off_or_fat(self):
        self.assertEqual(bc.resolve_compile({"lto": False})["lto"], "off")
        self.assertEqual(bc.resolve_compile({"lto": True})["lto"], "fat")
        for given, want in (("lto=off", "off"), ("lto=on", "fat"), ("lto=fat", "fat")):
            self.assertEqual(bc.resolve_compile(qf.parse_options([given]))["lto"], want, given)
        with self.assertRaisesRegex(ValueError, "--lto 'thin': give off or fat"):
            bc.resolve_compile({"lto": "thin"})

    def test_the_final_build_failing_names_the_lighter_build(self):
        fake = FakeRustc(self)

        def failing(cmd, env=None, watch=None):
            fake.cmds.append(cmd)
            return (1, "out of memory", 1.0) if any(a.startswith("profile-use=") for a in cmd) else fake.run_measured(cmd, env, watch)

        with mock.patch.object(bc, "run_measured", failing), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(bc.BuildFailed, r"rustc failed .*out of memory.*try --compile-preset 1.0"):
                bc.CodegenPgoBackend().compile(self.f, self.tmp / "no.so", {"pgo_rows": str(self.rows), "ram_mb": 100_000, "jobs": 1})
        with (
            mock.patch.object(bc, "run_measured", lambda cmd, env=None, watch=None: (1, "boom", 1.0)),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(bc.BuildFailed) as e:
                bc.CodegenBlockBackend().compile(self.f, self.tmp / "no.so", {"compile_preset": "1.0", "ram_mb": 100_000, "jobs": 1})
        self.assertNotIn("try --compile-preset", str(e.exception), "QF 1.0's build has no lighter build to name")

    def test_pgo_on_with_too_few_rows_raises_the_profile_line_and_best_still_falls_back(self):
        unreliable = bc.ProfileUnreliable("only 10 profile rows")
        out = self.tmp / "fallback.so"
        with (
            mock.patch.object(qf_compile, "best_backend", lambda *a, **k: ("codegen-pgo", None)),
            mock.patch.object(qf_compile, "profile_rows", lambda *a, **k: None),
            mock.patch.object(bc.CodegenPgoBackend, "compile", side_effect=unreliable),
            mock.patch.object(bc.CodegenBlockBackend, "compile", return_value={"built": "block"}),
        ):
            with self.assertRaisesRegex(bc.ProfileUnreliable, "only 10 profile rows"):
                qf_compile.compile_model(MODEL, out, "best", data="d.npz", pgo="on")
            info = qf_compile.compile_model(MODEL, out, "best", data="d.npz")
        self.assertEqual(info["backend"], "codegen-block")
        self.assertIn("only 10 profile rows", info["notice"])

    def test_pgo_on_is_refused_where_best_would_have_built_the_flat_format(self):
        f = forest()
        with mock.patch.object(bc, "available_ram_mb", lambda options: options.get("ram_mb")):
            with mock.patch.object(bc, "toolchain_bin", side_effect=bc.ToolchainMissing("rustc not found")):
                self.assertEqual(qf_compile.best_backend(f, True)[0], "flat")
                with self.assertRaisesRegex(ValueError, "--pgo on needs generated code.*rustc not found"):
                    qf_compile.best_backend(f, True, {"pgo": "on"})
            with mock.patch.object(bc, "toolchain_bin", lambda: Path("/fake/bin")):
                self.assertEqual(qf_compile.best_backend(f, True, {"ram_mb": 5})[0], "flat")
                with self.assertRaisesRegex(ValueError, "--pgo on needs generated code.*too little to build generated code"):
                    qf_compile.best_backend(f, True, {"ram_mb": 5, "pgo": "on"})

    def test_a_chunks_build_asks_the_ram_plan_for_the_lto_link_but_not_for_lto_chunks(self):
        chunks, one = bc.resolve_compile({}), bc.resolve_compile({"unit": "single"})
        self.assertLess(bc.peak_factor(chunks), bc.peak_factor(one))
        self.assertEqual(bc.link_peak_factor(chunks), bc.peak_factor(one))
        _, info = self.build(bc.CodegenBlockBackend)
        self.assertEqual(info["peak_factor"], round(bc.peak_factor(chunks), 2))

    def test_parent_site_paths_survives_a_site_module_without_getsitepackages(self):
        import site

        home = str(Path(np.__file__).resolve().parent.parent)
        saved = site.getsitepackages
        del site.getsitepackages
        try:
            self.assertIn(home, bc.parent_site_paths())
        finally:
            site.getsitepackages = saved

    def test_the_compare_worker_gets_the_parents_site_folders_under_python_I(self):
        """`python -I -S` has no site-packages at all: the worker finds numpy only if the folders are passed after `--worker SPEC`."""
        script = str(HERE.parent / "scripts" / "compare.py")
        run = lambda *extra: subprocess.run(  # noqa: E731
            [sys.executable, "-I", "-S", script, "--worker", str(self.tmp / "no-such-spec.json"), *extra], capture_output=True, text=True
        )
        self.assertIn("No module named 'numpy'", run().stderr)
        got = run(json.dumps(bc.parent_site_paths()))
        self.assertNotIn("No module named 'numpy'", got.stderr)
        self.assertIn("no-such-spec.json", got.stderr)  # it got as far as reading the spec


@unittest.skipUnless(shutil.which("rustc") or os.environ.get("RUST_DIR"), "needs rustc")
class ToySmokeTest(unittest.TestCase):
    """One real build of the toy forest at one job (about 20 s: fat LTO also optimises the standard library): the default's flags are accepted by
    rustc, the library loads and answers exactly as QF 1.0's build does."""

    def test_the_default_build_answers_as_the_1_0_build_does(self):
        try:
            bc.toolchain_bin()
        except bc.ToolchainMissing as e:
            self.skipTest(str(e))
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-toy-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        f, x = forest(), np.load(FIX / "x.npy")
        out = {}
        for tag, opts in (("old", {"compile_preset": "1.0"}), ("new", {})):
            with contextlib.redirect_stderr(io.StringIO()):
                bc.CodegenBlockBackend().compile(f, tmp / f"{tag}.so", {"jobs": 1, **opts})
            out[tag] = bc.CodegenForest(tmp / f"{tag}.so").predict(x, 1)
        self.assertTrue(np.array_equal(out["old"].view(np.uint32), out["new"].view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
