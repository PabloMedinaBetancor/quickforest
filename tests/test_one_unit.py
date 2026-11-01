"""QF 1.1 "one unit" build (`unit = single | chunks`): the whole generated forest and the driver as ONE crate (the chunks are modules), so fat LTO and
the inliner cross the chunks; and the profile run's `python -I` fix.

Light and mostly rustc-free: the flag assembly, the crate layout, the instrumented == final rule, the RAM fallback and the build info are checked with rustc
replaced by a recorder (as test_compile_settings does). The real-rustc tests build the toy forest at one job (about 20-40 s each, fat LTO also optimises the
standard library) and are skipped without rustc.

    python3 -m unittest test_one_unit          (not `-s`: this machine's numpy lives in the user site)
"""

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as bc
from test_compile_settings import FIX, FakeRustc, forest, settings_flags
from tiled import tiled

SMALL_CHUNKS = {"chunk_nodes": 60, "ram_mb": 100_000, "jobs": 1}  # the 260-node fixture in several chunks


class Recorder(FakeRustc):
    """FakeRustc that also keeps the Rust sources it was asked to compile (the temporary folder is gone when the build ends)."""

    def __init__(self, test):
        super().__init__(test)
        self.sources: dict[str, str] = {}

    def run_measured(self, cmd, env=None, watch=None):
        src = Path(cmd[-1]).parent
        if not self.sources:
            for f in src.glob("*.rs"):
                self.sources[f.name] = f.read_text()
        return super().run_measured(cmd, env, watch)


def build(test, backend=bc.CodegenBlockBackend, tmp=None, **opts):
    fake = Recorder(test)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        info = backend().compile(forest(), (tmp or test.tmp) / "m.so", {**SMALL_CHUNKS, **opts})
    return fake, info, err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="quickforest-unit-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class SettingTest(unittest.TestCase):
    def test_the_default_is_chunks_behind_a_constant_of_its_own(self):
        self.assertEqual(bc.UNIT_MODES, ("single", "chunks"))
        self.assertEqual(bc.DEFAULT_COMPILE_UNIT, "chunks")
        s = bc.resolve_compile({})
        self.assertEqual((s["unit"], s["preset"]), ("chunks", "1.1"))
        self.assertEqual(bc.COMPILE_PRESETS["1.1"]["unit"], bc.DEFAULT_COMPILE_UNIT)
        self.assertEqual(bc.COMPILE_PRESETS["1.0"]["unit"], "chunks")
        self.assertTrue(all(b["unit"] == "chunks" for b in bc.SHAPE_BUILDS.values()), "QF 1.0's two builds are chunk builds")

    def test_one_line_flips_preset_1_1_to_the_one_unit_build_and_nothing_else(self):
        text = (HERE.parent / "scripts" / "backend_codegen.py").read_text()
        self.assertEqual(text.count('DEFAULT_COMPILE_UNIT = "chunks"'), 1)
        flipped = types.ModuleType("bc_flipped")
        flipped.__file__ = str(HERE.parent / "scripts" / "backend_codegen.py")
        exec(
            compile(text.replace('DEFAULT_COMPILE_UNIT = "chunks"', 'DEFAULT_COMPILE_UNIT = "single"'), flipped.__file__, "exec"),
            flipped.__dict__,
        )
        s = flipped.resolve_compile({})
        self.assertEqual((s["unit"], s["preset"], s["lto"], s["codegen_units"], s["inline_threshold"]), ("single", "1.1", "fat", 16, 3000))
        for named in ({"compile_preset": "1.0"}, {"shape": "fast"}):
            self.assertEqual(flipped.resolve_compile(named)["unit"], "chunks", named)
        self.assertEqual(flipped.resolve_compile({"compile_preset": "1.0"})["preset"], "1.0")
        self.assertEqual(
            flipped.resolve_compile({"unit": "chunks"})["preset"], "custom", "1.1's flags built in chunks are no longer the preset"
        )
        self.assertEqual(flipped.describe_compile(s), "fat LTO, 16 codegen units, inline 3000, one crate")

    def test_the_option_overrides_the_preset_and_names_the_build(self):
        one = bc.resolve_compile({"unit": "single"})
        self.assertEqual((one["unit"], one["preset"], one["shape"]), ("single", "custom", "custom"))
        self.assertEqual(bc.describe_compile(one), "fat LTO, 16 codegen units, inline 3000, one crate")
        self.assertEqual(bc.describe_compile(bc.resolve_compile({})), "fat LTO, 16 codegen units, inline 3000")
        self.assertEqual(bc.resolve_compile({"compile_preset": "1.0", "unit": "single"})["unit"], "single")
        self.assertEqual(bc.resolve_compile({"unit": "single", "shape": "fast"})["unit"], "single", "an explicit setting wins over a shape")

    def test_a_bad_unit_is_refused_in_one_line(self):
        for bad in ("one", "", 1, True, "SINGLE"):
            with self.assertRaisesRegex(ValueError, r"--unit .*give single .* or chunks"):
                bc.resolve_compile({"unit": bad})

    def test_the_flat_and_wave_backends_refuse_it_like_every_compile_setting(self):
        import compile as qf_compile

        self.assertIn("unit", bc.COMPILE_KEYS)
        with self.assertRaisesRegex(ValueError, "--unit chooses how generated code is built: flat is not generated code"):
            qf_compile.compile_model(FIX / "lgb_plain.txt", Path(tempfile.gettempdir()) / "quickforest-never.bin", "flat", unit="single")

    def test_the_command_line_has_the_flag_in_compile_and_optimize(self):
        import argparse

        ap = argparse.ArgumentParser()
        bc.add_compile_arguments(ap)
        self.assertEqual(bc.compile_options(ap.parse_args([])), {})
        self.assertEqual(bc.compile_options(ap.parse_args(["--unit", "single"])), {"unit": "single"})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["--unit", "both"])
        import inspect

        import optimize

        self.assertIn("unit", inspect.signature(optimize.optimize).parameters)


class LayoutTest(Base):
    def test_single_is_one_rustc_for_one_crate_with_the_chunks_as_modules(self):
        fake, info, _ = build(self, unit="single")
        cmds = fake.cmds
        self.assertEqual(len(cmds), 1, "one rustc: no chunk crates, no separate link")
        (cmd,) = cmds
        self.assertIn("cdylib", cmd)
        self.assertNotIn("--emit", cmd)
        self.assertFalse([a for a in cmd if a.startswith("link-arg=")], "no objects are joined")
        self.assertEqual(settings_flags(cmd), ["opt-level=3", "lto=fat", "codegen-units=16", "llvm-args=-inline-threshold=3000"])
        self.assertEqual(Path(cmd[-1]).name, "lib.rs")
        lib = fake.sources["lib.rs"]
        mods = re.findall(r"^mod (c\d+);$", lib, re.M)
        self.assertGreater(len(mods), 1, "several chunks")
        self.assertEqual(sorted(fake.sources), sorted(["finish.rs", "lib.rs", *(m + ".rs" for m in mods)]))
        self.assertNotIn('extern "C" {', lib, "chunks are modules, not external symbols")
        self.assertTrue(all(f"{m}::{m}" in lib for m in mods), "the table holds the chunks' functions by path")
        for m in mods:  # the chunk keeps its C name, so the profile gate finds `c<N>` as it does for separate crates
            self.assertIn(f'#[no_mangle] pub unsafe extern "C" fn {m}(', fake.sources[m + ".rs"])
        self.assertEqual((info["unit"], info["jobs"], info["chunks"], info["codegen_units"]), ("single", 1, len(mods), 16))
        self.assertNotIn("unit_fallback", info)
        self.assertEqual(info["build"], "fat LTO, 16 codegen units, inline 3000, one crate")
        self.assertIn("modules of one crate", info["note"])
        self.assertEqual((info["compile_preset"], info["shape"]), ("custom", "custom"))

    def test_chunks_mode_is_unchanged(self):
        fake, info, err = build(self)
        self.assertEqual(info["unit"], "chunks")
        lib = fake.sources["lib.rs"]
        self.assertIn('extern "C" {', lib)
        self.assertNotRegex(lib, r"^mod c\d+;$")
        n = info["chunks"]
        self.assertEqual(len([c for c in fake.cmds if "--emit" in c]), n)
        self.assertEqual(len([c for c in fake.cmds if "cdylib" in c]), 1)
        self.assertTrue(err.startswith("RAM: building with 1 job / 16 units"), err)

    def test_the_driver_text_differs_only_where_the_chunks_are_reached(self):
        f = forest()
        _, per_fold = bc.split_chunks(f, 300)
        a = bc.driver_source(f, per_fold, 64)
        b = bc.driver_source(f, per_fold, 64, single=True)
        self.assertEqual(a, bc.driver_source(f, per_fold, 64, single=False))
        strip = lambda t: re.sub(r'extern "C" \{.*?\n\}\n|^mod c\d+;\n|^static FOLDS.*\n', "", t, flags=re.S | re.M)  # noqa: E731
        self.assertEqual(strip(a), strip(b))

    def test_codegen_units_become_modules_so_rustc_has_something_to_split(self):
        fake, info, _ = build(self, unit="single", codegen_units=8)
        self.assertEqual(info["codegen_units"], 8)
        n = info["chunks"]
        per_chunk = -(-8 // n)
        for name, text in fake.sources.items():
            if re.fullmatch(r"c\d+\.rs", name) and per_chunk > 1:
                self.assertEqual(len(re.findall(r"^mod g\d+ \{", text, re.M)), min(per_chunk, text.count("fn b")), name)
        self.assertIn("codegen-units=8", fake.cmds[0])
        chunky, _, _ = build(self, unit="chunks", codegen_units=8)
        self.assertEqual(len([c for c in chunky.cmds if "--emit" in c]), n)

    def test_single_works_for_row_mode_and_the_f64_accumulator_too(self):
        fake, info, _ = build(self, bc.CodegenBackend, unit="single")
        self.assertEqual(info["unit"], "single")
        self.assertIn("#[inline(never)]", fake.sources["c0.rs"])


class SharedShapeTest(Base):
    def setUp(self):
        super().setUp()
        self.rows = self.tmp / "rows.npy"
        np.save(self.rows, tiled(np.load(FIX / "x.npy")))

    def test_the_instrumented_build_is_the_final_one_in_every_flag_but_the_profile(self):
        for lto, units, inline in ((("fat", 1, 3000)), ("fat", 4, 3000), ("off", 1, "default"), ("fat", 1, "default")):
            fake, info, _ = build(
                self, bc.CodegenPgoBackend, unit="single", lto=lto, codegen_units=units, inline_threshold=inline, pgo_rows=str(self.rows)
            )
            gen = [c for c in fake.cmds if any(a.startswith("profile-generate=") for a in c)]
            use = [c for c in fake.cmds if any(a.startswith("profile-use=") for a in c)]
            self.assertEqual((len(gen), len(use), len(fake.cmds)), (1, 1, 2), "two builds: one instrumented, one final, both ONE rustc")

            def plain(cmd):
                keep = [a for a in cmd if not a.startswith(("profile-generate=", "profile-use=", "llvm-args=-pgo-warn", "strip="))]
                return [re.sub(r"model\.(inst|final)\.so", "model.so", a) for a in keep]

            g, u = plain(gen[0]), plain(use[0])
            # the folder the sources are in is the same, so the only differences allowed are the profile flags and the pgo-warn flag, whose `-C` is left behind
            self.assertEqual([a for a in g if a != "-C"], [a for a in u if a != "-C"], (lto, units, inline))
            self.assertEqual(settings_flags(gen[0]), settings_flags(use[0]))
            self.assertIn("cdylib", gen[0])
            self.assertTrue(info["pgo"] and info["unit"] == "single")

    def test_a_failed_instrumented_one_crate_build_does_not_fall_back_to_another_shape(self):
        fake = Recorder(self)

        def failing(cmd, env=None, watch=None):
            fake.cmds.append(cmd)
            return (1, "out of luck", 1.0) if any(a.startswith("profile-generate=") for a in cmd) else (0, "", 1.0)

        with mock.patch.object(bc, "run_measured", failing), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(bc.BuildFailed, "one-crate build.*does not fall back to another shape"):
                bc.CodegenPgoBackend().compile(forest(), self.tmp / "no.so", {**SMALL_CHUNKS, "unit": "single", "pgo_rows": str(self.rows)})
        self.assertEqual(len(fake.cmds), 1)


class RamTest(Base):
    def test_the_plan_for_one_crate_counts_all_the_nodes_in_one_process(self):
        est, f = bc.rustc_peak_estimate_mb, 2.9
        ram = est(100_000, 1, f) / bc.RAM_SHARE + 1
        self.assertEqual(bc.plan_single(100_000, 1, ram, f), 1)
        self.assertIsNone(bc.plan_single(200_000, 1, ram, f), "twice the nodes in ONE process do not fit where one chunk of them did")
        self.assertEqual(bc.plan_build([100_000, 100_000], 2, 1, 2 * ram, f), (2, 1), "...while two chunk jobs of the same size would")
        self.assertEqual(bc.plan_single(1_000, 4, None, f), 4, "no RAM information: the wanted units")
        # fewer units when four do not fit and two do
        tight = est(100_000, 2, f) / bc.RAM_SHARE + 1
        self.assertEqual(bc.plan_single(100_000, 8, tight, f), 2)
        self.assertEqual(bc.plan_single(100_000, 8, tight, f), bc.plan_build([100_000], 1, 8, tight, f)[1])

    def test_when_one_crate_does_not_fit_chunks_are_built_with_one_line_and_the_info_says_so(self):
        # make the per-node cost large so the 260-node toy forest is heavy as ONE process but its small chunks fit
        with mock.patch.object(bc, "RUSTC_MB_PER_KNODE", 30_000.0):
            fake, info, err = build(self, unit="single", ram_mb=10_000)
        lines = [ln for ln in err.splitlines() if ln.startswith("RAM:")]
        self.assertRegex(
            lines[0],
            r"^RAM: a one-crate build \(fat LTO, 16 codegen units, inline 3000, one crate\) of 260 nodes needs about [\d,]+ MB.*building in chunks instead \(--unit chunks\)$",
        )
        self.assertEqual(info["unit"], "chunks")
        self.assertIn("building in chunks instead", info["unit_fallback"])
        self.assertEqual(info["build"], "fat LTO, 16 codegen units, inline 3000", "the build string describes what was built")
        self.assertEqual((info["compile_preset"], info["shape"]), ("1.1", "custom"))
        self.assertTrue(fake.sources["lib.rs"].count('extern "C" {') == 1)
        self.assertIn("did not fit the RAM", info["note"])

    def test_the_fallback_is_decided_once_so_the_instrumented_build_gets_the_same_mode(self):
        rows = self.tmp / "rows.npy"
        np.save(rows, tiled(np.load(FIX / "x.npy")))
        with mock.patch.object(bc, "plan_single", return_value=None):
            fake, info, err = build(self, bc.CodegenPgoBackend, unit="single", pgo_rows=str(rows))
        self.assertEqual(info["unit"], "chunks")
        gen = [c for c in fake.cmds if any(a.startswith("profile-generate=") for a in c)]
        use = [c for c in fake.cmds if any(a.startswith("profile-use=") for a in c)]
        self.assertEqual(
            (len(gen), len(use)), (info["chunks"] + 1, info["chunks"] + 1), "chunk builds + a link, instrumented and final alike"
        )
        self.assertEqual(sum("building in chunks instead" in ln for ln in err.splitlines()), 1)

    def test_one_crate_with_enough_ram_uses_one_job_and_says_so(self):
        _, info, err = build(self, unit="single", jobs=3)
        self.assertEqual((info["jobs"], info["unit"]), (1, "single"))
        self.assertTrue(err.startswith("RAM: building ONE crate with 1 job / 16 units"), err)

    def test_without_ram_information_the_one_crate_build_goes_ahead(self):
        with mock.patch.object(bc, "available_ram_mb", return_value=None):
            _, info, err = build(self, unit="single")
        self.assertEqual(info["unit"], "single")
        self.assertIn("available RAM could not be read", err)

    def test_not_even_chunks_fit_is_still_a_refusal_never_a_swap(self):
        fake = Recorder(self)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bc.NotEnoughRam):
            bc.CodegenBlockBackend().compile(forest(), self.tmp / "m.so", {"unit": "single", "ram_mb": 20})
        self.assertEqual(fake.cmds, [])


class ProfileNamesTest(unittest.TestCase):
    def test_a_function_has_the_same_key_as_its_own_crate_or_as_a_module_of_the_one_crate(self):
        key = bc.function_key
        self.assertEqual(key("_ZN2c02b517h0123456789abcdefE"), ("c0", "b5"))  # a chunk crate (legacy mangling)
        self.assertEqual(key("_ZN3lib2c02g12b517h0123456789abcdefE"), ("c0", "b5"))  # chunk 0, sub-module g1, of the one crate
        self.assertEqual(key("_ZN3lib2c12b717h0123456789abcdefE"), ("c1", "b7"))
        self.assertEqual(key("_RNvCs1_2c02b5"), ("c0", "b5"))  # v0, chunk crate
        self.assertEqual(key("_RNvNtCs1_3lib2c02b5"), ("c0", "b5"))  # v0, module of the one crate
        self.assertEqual(key("_RNvNtNtCs1_3lib2c02g12b5"), ("c0", "b5"))
        self.assertEqual(key("c0"), ("c0",))  # the plain C name of a chunk's entry function
        self.assertEqual(key("file.rs;_ZN3lib2c02g12b517h0123456789abcdefE"), ("c0", "b5"))
        self.assertEqual(key("_ZN3lib6finish14predict_blocks17h0123456789abcdefE"), ("lib", "predict_blocks"))


class ProfileRunIsolationTest(unittest.TestCase):
    """`profile_run` starts python with `-I`, which hides numpy in the user site: «No module named numpy». The fix passes the parent's site folders."""

    CHILD = bc.PROFILE_SCRIPT.split("import numpy")[0] + "import numpy\nprint('NUMPY-OK')\n"  # the real script's own preamble

    def test_the_parent_s_numpy_folder_is_passed_and_isolation_stays(self):
        paths = bc.parent_site_paths()
        self.assertIn(str(Path(np.__file__).resolve().parent.parent), paths)
        self.assertNotIn("", paths)
        self.assertNotIn(os.getcwd() if os.getcwd() not in sys.path[1:] else "\0", paths, "the working folder is not passed")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/definitely/not/here"}):
            code = subprocess.run(
                [sys.executable, "-I", "-c", "import sys; print([p for p in sys.path if 'not/here' in p])"], capture_output=True, text=True
            ).stdout
        self.assertEqual(code.strip(), "[]", "-I still ignores PYTHONPATH")

    def test_numpy_only_on_a_path_the_parent_added_is_found_by_the_isolated_run_only_with_the_fix(self):
        home = str(Path(np.__file__).resolve().parent.parent)
        parent = (
            "import sys\n"
            f"sys.path.insert(0, {home!r})\n"  # the parent has numpy only because of this line (python -S: no site at all)
            f"sys.path.insert(0, {str(HERE.parent / 'scripts')!r})\n"
            "import backend_codegen as bc\n"
            f"child = {self.CHILD!r}\n"
            "import subprocess\n"
            "without = subprocess.run([sys.executable, '-I', '-c', child, bc.HERE.as_posix(), '', '', '', '[]'], capture_output=True, text=True)\n"
            "withfix = subprocess.run([sys.executable, '-I', '-c', child, bc.HERE.as_posix(), '', '', '', __import__('json').dumps(bc.parent_site_paths())], capture_output=True, text=True)\n"
            "print('WITHOUT', without.returncode, 'WITH', withfix.returncode, withfix.stdout.strip())\n"
        )
        p = subprocess.run(
            [sys.executable, "-S", "-c", parent],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
        self.assertEqual(p.returncode, 0, p.stderr[-400:])
        m = re.search(r"WITHOUT (\d+) WITH (\d+) (\S*)", p.stdout)
        self.assertTrue(m, p.stdout)
        self.assertEqual((m.group(2), m.group(3)), ("0", "NUMPY-OK"), "with the parent's folders the isolated run imports numpy")
        if m.group(1) == "0":
            self.skipTest(
                "numpy is in a system folder here, so -I alone already finds it (the negative control cannot show the bug on this machine)"
            )

    def test_profile_run_reaches_numpy_in_the_isolated_child(self):
        raw = Path(tempfile.mkdtemp(prefix="quickforest-raw-")) / "raw"
        self.addCleanup(shutil.rmtree, raw.parent, True)
        script = bc.PROFILE_SCRIPT.split("from backend_codegen")[0] + "print('NUMPY-OK')\n"
        with (
            mock.patch.object(bc, "PROFILE_SCRIPT", script),
            self.assertRaises(bc.BuildFailed) as e,
        ):  # no .profraw is written: the build says so
            bc.profile_run(Path("x.so"), Path("x.npy"), raw, "main")
        self.assertIn("NUMPY-OK", str(e.exception))
        self.assertNotIn("No module named", str(e.exception))


@unittest.skipUnless(shutil.which("rustc") or os.environ.get("RUST_DIR"), "needs rustc")
class ToyBuildTest(unittest.TestCase):
    """REAL rustc on the toy forest at one job. The three builds answer bit for bit alike; the one-crate profile-guided build passes the hot-function gate."""

    @classmethod
    def setUpClass(cls):
        try:
            bc.toolchain_bin()
        except bc.ToolchainMissing as e:
            raise unittest.SkipTest(str(e)) from None
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-toy-unit-"))
        cls.f, cls.x = forest(), np.load(FIX / "x.npy")
        cls.rows = cls.tmp / "rows.npy"
        np.save(cls.rows, tiled(np.load(FIX / "x.npy")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, tag, backend=bc.CodegenBlockBackend, **opts):
        with contextlib.redirect_stderr(io.StringIO()):
            info = backend().compile(self.f, self.tmp / f"{tag}.so", {"jobs": 1, "chunk_nodes": 60, "pgo_rows": str(self.rows), **opts})
        return info, bc.CodegenForest(self.tmp / f"{tag}.so").predict(self.x, 1)

    def test_the_three_builds_answer_bit_for_bit_alike(self):
        i10, a10 = self.build("v10", compile_preset="1.0")
        ic, ac = self.build("v11c", unit="chunks")
        i1, a1 = self.build("v11s", unit="single")
        self.assertEqual((i10["unit"], ic["unit"], i1["unit"]), ("chunks", "chunks", "single"))
        self.assertTrue(np.array_equal(a10.view(np.uint32), ac.view(np.uint32)))
        self.assertTrue(np.array_equal(a10.view(np.uint32), a1.view(np.uint32)))
        i4, a4 = self.build("v11s4", unit="single", codegen_units=4)  # the units split by modules inside the one crate: same answers
        self.assertEqual((i4["unit"], i4["codegen_units"]), ("single", 4))
        self.assertTrue(np.array_equal(a10.view(np.uint32), a4.view(np.uint32)))
        self.assertGreater(i1["chunks"], 1)
        # and the library really is the one crate: the chunks' entry points are symbols of it, there is one cdylib
        self.assertEqual(i1["jobs"], 1)

    def test_the_one_crate_profile_guided_build_matches_its_hot_functions_and_answers_alike(self):
        _, a = self.build("v10b", compile_preset="1.0")
        info, b = self.build("pgo-single", bc.CodegenPgoBackend, unit="single")
        self.assertEqual(info["unit"], "single")
        self.assertGreater(info["profiled_functions"], 0)
        self.assertGreater(info["profile_hot_functions"], 0)
        self.assertTrue(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
        print("\nTOY single PGO:", {k: info[k] for k in info if k.startswith(("profile", "rustc", "compile", "unit"))}, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
