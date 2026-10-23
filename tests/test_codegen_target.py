"""The generated-code path builds the fastest portable code and says so: opt-level 3 for the x86-64-v2 CPU by default (`--native` is the opt-in
for the machine it is built on), the CPU features written INTO the library and checked in Python before the library is loaded, a profile that
must have been applied to every hot generated function, and `best` (profile-guided with data, `codegen-block` without, flat only where generated
code cannot be built) as the default of compile and optimize. Needs rustc; each build is a model of a few dozen trees.

    python3 -s -m unittest test_codegen_target
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import backend_codegen as bc
import backends
import compile as qf_compile
import optimize
import qf_common as qf
import qf_formats
from tiled import tiled

FIX = HERE / "fixtures"
BITS = np.uint32
# Tests use the fastest compile: only the checks of the defaults themselves build with the defaults (opt-level 3, x86-64-v2,
# profile-guided); every other build here asks for opt-level 0, whose answers must be the same bit for bit.
FAST = {"opt_level": 0}


def forest(name: str):
    return qf.load_forest(qf_formats.model_files(FIX / name), "output", False)


def same_bits(a, b) -> bool:
    return np.array_equal(np.ascontiguousarray(a, np.float32).view(BITS), np.ascontiguousarray(b, np.float32).view(BITS))


def cpuinfo(path: Path, *cpus: set) -> Path:
    """A made-up /proc/cpuinfo: one block per logical CPU with the given features (x86 `flags`, arm `Features`)."""
    key = "flags" if bc.ARCH in ("x86_64", "amd64") else "Features"
    path.write_text("\n".join(f"processor\t: {i}\n{key}\t\t: {' '.join(sorted(c))}\n" for i, c in enumerate(cpus)))
    return path


class TargetCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-target-"))
        cls.f = forest("xgb_multi.ubj")
        cls.x = np.load(FIX / "x.npy")
        cls.info, cls.so = {}, {}
        widest = "x86-64" if bc.default_cpu() == "x86-64-v2" else "generic"
        # the defaults (one build), the opt-in for this machine, and the widest CPU of the architecture
        for tag, opts in (("default", {}), ("native", {**FAST, "target_cpu": "native"}), ("wide", {**FAST, "target_cpu": widest})):
            cls.so[tag] = cls.tmp / f"{tag}.so"
            cls.info[tag] = bc.CodegenBlockBackend().compile(cls.f, cls.so[tag], opts)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def checkable(self, tag):
        """The recorded features this CPU check can tell from /proc/cpuinfo."""
        return [f for f in bc.read_target(self.so[tag])[2] if f in bc.CPUINFO_FLAGS[bc.ARCH]]

    def test_every_generated_code_backend_builds_at_opt_level_3(self):
        for be in (bc.CodegenBackend(), bc.CodegenBlockBackend(), bc.CodegenPgoBackend()):
            self.assertEqual(be.opt_level, 3, be.name)
        self.assertIn("opt 3", self.info["default"]["note"])
        self.assertIn(f"shape {bc.DEFAULT_SHAPE}", self.info["default"]["note"])
        self.assertIn(f"{bc.SHAPES[bc.DEFAULT_SHAPE]} codegen units", self.info["default"]["note"])

    def test_the_default_is_the_portable_cpu_and_the_result_names_it(self):
        self.assertEqual(bc.default_cpu(), "x86-64-v2" if bc.ARCH in ("x86_64", "amd64") else "generic")
        self.assertEqual(self.info["default"]["target_cpu"], bc.default_cpu())
        self.assertIn(f"target {bc.default_cpu()}", self.info["default"]["note"])
        cpu, arch, feats = bc.read_target(self.so["default"])
        self.assertEqual((cpu, arch), (bc.default_cpu(), bc.ARCH))
        self.assertEqual(len(feats), self.info["default"]["target_features"])
        self.assertEqual(bc.CodegenForest(self.so["default"]).target_cpu, bc.default_cpu())
        if bc.default_cpu() == "x86-64-v2":
            self.assertIn("sse4.2", feats)
            self.assertFalse({"avx", "avx2", "fma", "bmi2", "avx512f"} & set(feats), "AVX is never compiled in by default")

    def test_native_is_an_opt_in_that_records_the_features_of_this_machine(self):
        self.assertEqual(self.info["native"]["target_cpu"], "native")
        self.assertIn("target native", self.info["native"]["note"])
        cpu, arch, feats = bc.read_target(self.so["native"])
        self.assertEqual((cpu, arch), ("native", bc.ARCH))
        self.assertEqual(feats, bc.target_features(str(bc.toolchain_bin() / "rustc"), "native"))
        self.assertEqual(len(feats), self.info["native"]["target_features"])
        self.assertTrue(feats)
        if bc.default_cpu() == "x86-64-v2":
            self.assertLessEqual(set(bc.read_target(self.so["default"])[2]), set(feats))
        self.assertEqual(bc.CodegenForest(self.so["native"]).target_cpu, "native")

    def test_the_two_spellings_of_native(self):
        self.assertEqual(bc.cpu_option(True, None), "native")
        self.assertEqual(bc.cpu_option(True, "native"), "native")
        self.assertEqual(bc.cpu_option(False, "x86-64-v3"), "x86-64-v3")
        self.assertIsNone(bc.cpu_option(False, None))
        with self.assertRaisesRegex(ValueError, "alternatives"):
            bc.cpu_option(True, "x86-64-v3")

    def test_the_answers_do_not_depend_on_the_cpu_built_for(self):
        """Rust does not fuse a multiply and an add without fast-math, so the default build (opt-level 3) is bit-identical to the native one
        and to the widest one built at opt-level 0."""
        want = bc.CodegenForest(self.so["default"]).predict(self.x, 1)
        for tag in ("native", "wide"):
            self.assertTrue(same_bits(bc.CodegenForest(self.so[tag]).predict(self.x, 1), want), tag)

    def test_native_is_bit_identical_on_a_lightgbm_model(self):
        f, out = forest("lgb_plain.txt"), []
        for tag, opts in (("default", {}), ("native", {**FAST, "target_cpu": "native"})):
            so = self.tmp / f"lgb-{tag}.so"
            bc.CodegenBlockBackend().compile(f, so, opts)
            out.append(bc.CodegenForest(so).predict(self.x, 1))
        self.assertTrue(same_bits(*out))

    def test_a_library_for_features_this_cpu_lacks_is_refused_before_it_is_loaded(self):
        feats = self.checkable("native")
        self.assertTrue(feats, "the native build records at least one feature the check knows")
        names = bc.CPUINFO_FLAGS[bc.ARCH]
        old_cpu = {names[f] for f in feats[1:]}  # every feature but the first one
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "old.txt", old_cpu)), mock.patch.object(bc.ctypes, "CDLL") as dlopen:
            with self.assertRaises(bc.TargetCpuMismatch) as cm:
                bc.CodegenForest(self.so["native"])
            dlopen.assert_not_called()  # the library never got to run
        msg = str(cm.exception)
        self.assertIn(feats[0], msg)
        self.assertIn("--native", msg)
        self.assertNotIn("\n", msg)
        self.assertIsInstance(cm.exception, ValueError)  # every command prints a ValueError as one line and exits 3

    def test_the_profile_pass_is_refused_before_the_first_build_on_a_cpu_that_cannot_run_the_code(self):
        """codegen-pgo RUNS the instrumented code it builds: a target this CPU lacks must end in one line, not in a traceback after a build."""
        rows = self.tmp / "rows.npy"
        np.save(rows, self.x[:50])
        nothing = cpuinfo(self.tmp / "nothing.txt", set())
        with (
            mock.patch.object(bc, "CPUINFO", nothing),
            mock.patch.object(bc.tempfile, "mkdtemp", side_effect=AssertionError("a build was started")),
            self.assertRaises(bc.BuildFailed) as cm,
        ):
            bc.CodegenPgoBackend().compile(self.f, self.tmp / "pgo.so", {"pgo_rows": str(rows)})
        msg = str(cm.exception)
        self.assertEqual(len(msg.splitlines()), 1)
        self.assertIn("--backend codegen-block", msg)
        self.assertIn(self.checkable("default")[0], msg)
        self.assertFalse((self.tmp / "pgo.so").exists())

    def test_check_refuses_a_library_for_features_this_cpu_lacks_with_one_line_and_exit_3(self):
        import check

        err = io.StringIO()
        argv = ["check.py", str(FIX / "xgb_multi.ubj"), str(self.so["default"]), "--random", "20"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "nothing2.txt", set())),
            contextlib.redirect_stderr(err),
            self.assertRaises(SystemExit) as cm,
        ):
            check.main()
        self.assertEqual(cm.exception.code, 3)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("check.py: ") and "illegal instruction" in lines[0], lines[0])

    def test_the_default_library_is_checked_the_same_way(self):
        feats = self.checkable("default")
        names = bc.CPUINFO_FLAGS[bc.ARCH]
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "ancient.txt", {names[f] for f in feats[1:]})):
            with self.assertRaises(bc.TargetCpuMismatch) as cm:
                bc.CodegenForest(self.so["default"])
        self.assertIn(feats[0], str(cm.exception))

    def test_a_cpu_with_every_feature_loads_it(self):
        names = bc.CPUINFO_FLAGS[bc.ARCH]
        full = cpuinfo(self.tmp / "full.txt", {names[f] for f in self.checkable("native")})
        with mock.patch.object(bc, "CPUINFO", full):
            bc.CodegenForest(self.so["native"])

    def test_a_processor_that_lacks_a_feature_counts_for_the_machine(self):
        """A process may be scheduled on any logical CPU: the features of the machine are the ones ALL of them have."""
        names = bc.CPUINFO_FLAGS[bc.ARCH]
        feats = self.checkable("native")
        everything, less = {names[f] for f in feats}, {names[f] for f in feats[1:]}
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "hybrid.txt", everything, less)):
            self.assertEqual(bc.cpu_flags(), less)
            with self.assertRaises(bc.TargetCpuMismatch):
                bc.CodegenForest(self.so["native"])

    def test_the_default_library_loads_on_a_cpu_with_only_the_default_features(self):
        names = bc.CPUINFO_FLAGS[bc.ARCH]
        base = {names[f] for f in self.checkable("default")}
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "base.txt", base)):
            bc.CodegenForest(self.so["default"])

    def test_an_unreadable_cpu_list_does_not_refuse(self):
        with mock.patch.object(bc, "CPUINFO", self.tmp / "nosuchfile"):
            self.assertIsNone(bc.cpu_flags())
            bc.CodegenForest(self.so["native"])

    def test_a_library_without_the_record_loads_unchecked(self):
        """Libraries built before the record existed carry none: they load as they always did."""
        old = self.tmp / "old.so"
        old.write_bytes(self.so["native"].read_bytes().replace(b"QFTARGET1|", b"QFTARGET0|"))
        self.assertIsNone(bc.read_target(old))
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "none.txt", set())):
            self.assertEqual(bc.CodegenForest(old).target_cpu, None)

    def test_the_record_survives_a_stripped_library(self):
        """`strip=symbols` removes the symbol table, not the record in the data."""
        self.assertEqual(bc.read_target(self.so["native"])[1], bc.ARCH)

    def test_options_that_cannot_be_used_are_refused_before_anything_is_built(self):
        be, out = bc.CodegenBlockBackend(), self.tmp / "never.so"
        for opts, word in (
            ({"target_cpu": "nosuchcpu-v9"}, "does not know"),
            ({"target_cpu": "native; rm -rf x"}, "CPU name"),
            ({"target_cpu": 3}, "CPU name"),
        ):
            with self.subTest(opts), self.assertRaises(ValueError) as cm:
                be.compile(self.f, out, opts)
            self.assertIn(word, str(cm.exception))
            self.assertFalse(out.exists())

    def test_the_feature_names_the_check_knows_are_in_the_tables_of_both_architectures(self):
        for arch in ("x86_64", "aarch64"):
            self.assertIn("avx2" if arch == "x86_64" else "neon", bc.CPUINFO_FLAGS[arch])
        self.assertEqual(bc.CPUINFO_FLAGS["x86_64"]["lzcnt"], "abm")
        self.assertEqual(bc.CPUINFO_FLAGS["aarch64"]["crc"], "crc32")


class ProfileGateTest(unittest.TestCase):
    """rustc applies a profile that does not match the code silently (exit 0): the build must check that every hot generated function has one."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-gate-"))
        cls.rows = cls.tmp / "rows.npy"
        np.save(cls.rows, tiled(np.load(FIX / "x.npy")))
        cls.profile = {}
        cls.info = {}
        for name in ("xgb_multi.ubj", "xgb_reg.json"):
            so = cls.tmp / f"{name}.so"
            cls.info[name] = bc.CodegenPgoBackend().compile(forest(name), so, {**FAST, "pgo_rows": str(cls.rows), "keep_src": True})
            cls.profile[name] = Path(cls.info[name]["src_dir"]) / "merged.profdata"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build_with(self, name: str, profile: Path):
        """`name` built profile-guided, but with the given profile in place of the one its own profile run wrote."""

        def swap(merged, raw_files):
            shutil.copyfile(profile, merged)

        with mock.patch.object(bc, "merge_profiles", swap):
            return bc.CodegenPgoBackend().compile(forest(name), self.tmp / "gate.so", {**FAST, "pgo_rows": str(self.rows)})

    def test_a_matching_profile_covers_every_generated_function(self):
        f = forest("xgb_multi.ubj")
        # one function per tree, the chunk's entry and the driver's loop over row blocks
        self.assertEqual(self.info["xgb_multi.ubj"]["profiled_functions"], f.n_trees + 1 + 1)
        self.assertIn(f"profile on {f.n_trees + 2:,} functions", self.info["xgb_multi.ubj"]["note"])
        self.assertNotIn("profile_cold_unmatched", self.info["xgb_multi.ubj"], "a matching profile leaves no warning at all")

    def test_the_hot_functions_are_the_fewest_that_hold_90_percent_of_the_counts_of_a_real_profile(self):
        """The profile of the fixture, as `llvm-profdata show --all-functions --counts` of THIS machine prints it. These builds are opt-level 0, where the
        standard library's helpers are functions of their own and instrumented too, so the hot set is not just the trees: only its definition is
        checked, and a failure prints what was parsed."""
        f = forest("xgb_multi.ubj")
        table = bc.profile_table(self.profile["xgb_multi.ubj"])
        heaviest = sorted(table.items(), key=lambda kv: -kv[1][1])
        hot = bc.hot_functions(table)
        total, held = sum(w for _, w in table.values()), sum(table[n][1] for n in hot)
        detail = (
            f"hot {len(hot)} of {len(table)} profile functions (n_trees {f.n_trees}), they hold {held:,} of {total:,} counts; the 12 heaviest: "
            + "; ".join(f"{w:,} {n[:60]}" for n, (_, w) in heaviest[:12])
        )
        self.assertGreater(total, 0, detail)
        self.assertEqual(self.info["xgb_multi.ubj"]["profile_hot_functions"], len(hot), detail)
        self.assertGreaterEqual(len(hot), 1, detail)
        self.assertGreaterEqual(held, bc.HOT_SHARE * total, detail)
        self.assertLess(held - table[hot[-1]][1], bc.HOT_SHARE * total, "the fewest: without the last one it falls short; " + detail)
        self.assertEqual(hot, [n for n, _ in heaviest[: len(hot)]], "hottest first; " + detail)

    def test_the_profile_of_a_forest_with_fewer_trees_is_refused(self):
        """xgb_reg has 30 trees, xgb_multi 36: the 30 trees of the profile are the hot functions, and none of them has the control flow of the
        tree of that number in xgb_multi."""
        with self.assertRaises(bc.BuildFailed) as cm:
            self.build_with("xgb_multi.ubj", self.profile["xgb_reg.json"])
        self.assertRegex(str(cm.exception), r"dropped the profile of \d+ of its \d+ hottest functions")
        self.assertNotIn("\n", str(cm.exception))
        self.assertIsInstance(cm.exception, ValueError)  # one line, exit 3

    def test_a_profile_with_the_same_function_names_but_other_control_flow_is_refused(self):
        """The 30 trees of xgb_reg are all in the profile of xgb_multi by name, but their branches are not the profiled ones: rustc drops
        their counts with a warning only."""
        with self.assertRaises(bc.BuildFailed) as cm:
            self.build_with("xgb_reg.json", self.profile["xgb_multi.ubj"])
        self.assertRegex(str(cm.exception), r"dropped the profile of \d+ of its \d+ hottest functions|hottest functions are not in it")
        self.assertNotIn("\n", str(cm.exception))

    def test_the_gate_fails_on_the_hot_functions_by_counts_and_a_cold_miss_only_counts(self):
        """Hot = the fewest functions that hold 90 % of the profile's counts. LLVM also warns about code the profile run never executed or barely
        touched (an error path, a chunk function that ran 0 times): cold, counted and never fatal. A hot function that lost its profile, under its
        own name or under another codegen unit's, fails the build, and so does a warning this check cannot read."""
        h = "0123456789abcdef"
        tree = lambda k, cgu=1: f"c0.{h}-cgu.{cgu};_ZN2c02g02b{k}17h{h}E"  # noqa: E731  (a private function carries its object's name)
        weights = {tree(0): 3000, tree(1): 3000, tree(2): 3000, tree(3): 0, "c0": 60, f"_ZN3lib6finish14predict_blocks17h{h}E": 30}
        table = {n: ("0x1", w) for n, w in weights.items()}
        table[f"_ZN3lib6finish9cold_path17h{h}E"] = ("0x2", 0)
        plan = [(0, 0, [None] * 4)]  # check_profile reads only the number of trees of each chunk
        self.assertEqual(sorted(bc.hot_functions(table)), sorted([tree(0), tree(1), tree(2)]), "three trees hold 9,000 of 9,090 counts")

        def check(*lines, tab=table):
            watch = bc.Watch()
            for ln in lines:
                watch.add(ln)
            with mock.patch.object(bc, "profile_table", return_value=tab):
                return bc.check_profile(watch, Path("merged.profdata"), plan, "block")

        warn = lambda what, name: f"warning: c0.{h}-cgu.3: {what} {name} Hash = 594112148306129356 up to 0 count discarded"  # noqa: E731  (LLVM's own wording)
        nodata, mismatch = "no profile data available for function", "function control flow change detected (hash mismatch)"
        self.assertEqual(check(), {"generated": 4 + 1 + 1, "hot": 3, "cold": 0})
        cold = [  # the zero-count tree, the entry, the driver's loop and a function the profile does not hold: none of them is hot
            warn(nodata, f"_ZN2c02g02b3{'17h' + h}E"),
            warn(nodata, "c0"),
            warn(nodata, f"_ZN3lib6finish14predict_blocks17h{h}E"),
            warn(mismatch, f"_ZN3lib6finish9cold_path17h{h}E"),
            warn(nodata, f"_ZN3lib6finish5other17h{h}E"),
        ]
        self.assertEqual(check(*cold), {"generated": 6, "hot": 3, "cold": 5}, "cold misses pass and are counted")
        hot_misses = {
            "a hash mismatch on a hot tree": warn(mismatch, tree(2).split(";")[1]),
            "no data under another codegen unit's name": warn(nodata, tree(1, cgu=5).split(";")[1]),
            "a line it cannot read": "warning: function control flow change detected somewhere",
        }
        for what, bad in hot_misses.items():
            with self.assertRaises(bc.BuildFailed, msg=what) as cm:
                check(*cold, bad)
            self.assertIn("dropped the profile of 1 of its 3 hottest functions", str(cm.exception), what)
            self.assertNotIn("\n", str(cm.exception))
        # a profile whose hot trees are not all in this code (a bigger forest's), and a profile with no counts
        bigger = {**table, f"c0.{h}-cgu.1;_ZN2c02g02b9{'17h' + h}E": ("0x1", 5000)}
        with self.assertRaisesRegex(bc.BuildFailed, "hottest functions are not in it .first: c0::b9"):
            check(tab=bigger)
        for empty in ({}, {n: ("0x1", 0) for n in table}):
            with self.assertRaisesRegex(bc.BuildFailed, "the profile has no counts"):
                check(tab=empty)

    def test_symbol_names_of_a_function_in_a_module_give_the_crate_and_the_function(self):
        """The fast shape spreads a chunk's trees over modules: `c0::g3::b25`. The gate keys a function by its crate and its own name."""
        self.assertEqual(bc.mangled_names("c0.635adeaa32a6a23c-cgu.3;_RNvNtCs8wRzGhyUBXs_2c02g33b25"), ["c0", "g3", "b25"])
        self.assertEqual(bc.mangled_names("_ZN2c02g33b2517h0123456789abcdefE"), ["c0", "g3", "b25"])

    def test_symbol_names_of_both_manglings_give_the_crate_and_the_function(self):
        self.assertEqual(bc.mangled_names("c0.635adeaa32a6a23c-cgu.0;_RNvCs8wRzGhyUBXs_2c03b25"), ["c0", "b25"])
        self.assertEqual(bc.mangled_names("_RNvCs8wRzGhyUBXs_2c02b3"), ["c0", "b3"])
        self.assertEqual(bc.mangled_names("_ZN2c03b2517h0123456789abcdefE"), ["c0", "b25"])
        self.assertEqual(bc.mangled_names("tfl_cg_predict"), ["tfl_cg_predict"])

    def run_script(self, kind: str):
        """Runs the profile run's script on the finished library of the fixture and returns the calls it made: (rows, threads, margin, dtype)."""
        calls = []
        real = bc.CodegenForest.predict

        def spy(this, x, threads=None, base_margin=None, dtype=np.float32):
            calls.append((len(x), threads, base_margin is not None, np.dtype(dtype)))
            return real(this, x, threads, base_margin, dtype)

        so, rows = self.tmp / "xgb_multi.ubj.so", np.load(self.rows)
        with (
            mock.patch.object(bc.CodegenForest, "predict", spy),
            mock.patch.object(sys, "argv", ["-c", str(HERE.parent / "scripts"), str(so), str(self.rows), kind]),
        ):
            exec(compile(bc.PROFILE_SCRIPT, "profile-script", "exec"), {})
        return calls, len(rows)

    def test_the_halves_pass_is_one_batch_on_one_thread(self):
        calls, n = self.run_script("main")
        self.assertEqual(calls, [(n, 1, False, np.dtype(np.float32))])

    def test_the_coverage_pass_adds_two_threads_float64_answers_and_a_base_margin(self):
        calls, n = self.run_script("cover")
        self.assertEqual(
            calls,
            [(n, 2, False, np.dtype(np.float32)), (n, 1, False, np.dtype(np.float64)), (n, 1, True, np.dtype(np.float32))],
        )


class BuildShapeTest(unittest.TestCase):
    """The default build is the fast shape (opt 3, no LTO, 16 codegen units, the profile pass in the same shape); `top` is one unit; the jobs and
    units are planned against the free RAM and the build never swaps."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-shape-"))
        cls.f = forest("xgb_multi.ubj")
        cls.x = np.load(FIX / "x.npy")
        cls.pgo_rows = cls.tmp / "pgo-rows.npy"
        np.save(cls.pgo_rows, tiled(cls.x))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, tag, opts, backend=bc.CodegenBlockBackend):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            info = backend().compile(self.f, self.tmp / f"{tag}.so", {**FAST, **opts})
        return info, err.getvalue()

    def test_the_default_shape_is_fast(self):
        self.assertEqual(bc.DEFAULT_SHAPE, "fast")
        default, _ = self.build("default", {})
        self.assertEqual((default["shape"], default["codegen_units"]), (bc.DEFAULT_SHAPE, bc.SHAPES[bc.DEFAULT_SHAPE]))

    def test_fast_is_16_units_top_is_one_and_the_answers_are_the_same(self):
        self.assertEqual(bc.SHAPES, {"fast": 16, "top": 1})
        fast, _ = self.build("fast", {"shape": "fast"})
        top, _ = self.build("top", {"shape": "top"})
        self.assertEqual((fast["shape"], fast["codegen_units"]), ("fast", 16))
        self.assertEqual((top["shape"], top["codegen_units"]), ("top", 1))
        self.assertIn("shape top", top["note"])
        want = bc.CodegenForest(self.tmp / "top.so").predict(self.x, 1)
        got = bc.CodegenForest(self.tmp / "fast.so").predict(self.x, 1)
        self.assertTrue(same_bits(got, want), "the shape changes how the code is built, never its answers")

    def test_the_units_are_real_the_trees_sit_in_modules_and_rustc_makes_one_object_per_unit(self):
        """`-C codegen-units=16` splits nothing in a source of a single module, so the fast shape spreads the trees over modules."""
        fast, _ = self.build("fast-src", {"keep_src": True, "shape": "fast"})
        top, _ = self.build("top-src", {"keep_src": True, "shape": "top"})
        try:
            src_f, src_t = Path(fast["src_dir"]), Path(top["src_dir"])
            self.assertIn("mod g0 {", (src_f / "c0.rs").read_text())
            self.assertNotIn("mod g", (src_t / "c0.rs").read_text())
            self.assertGreater(len(bc.object_files(src_f / "o-final-c0")), 1)
            self.assertEqual(len(bc.object_files(src_t / "o-final-c0")), 1)
        finally:
            for r in (fast, top):
                shutil.rmtree(r["src_dir"], ignore_errors=True)

    def test_modules_hold_consecutive_trees_balanced_by_nodes_and_a_small_chunk_has_fewer(self):
        trees = self.f.models[0].trees
        home = bc.module_of_tree(trees, 4)
        self.assertEqual(len(home), len(trees))
        self.assertEqual(home, sorted(home), "consecutive trees share a module")
        self.assertEqual(set(home), {0, 1, 2, 3})
        src = bc.chunk_source(trees[:1], 0, 0, self.f.n_features, self.f.groups, "block", units=16)
        self.assertNotIn("mod g", src, "one tree needs no module")
        src = bc.chunk_source(trees[:3], 0, 0, self.f.n_features, self.f.groups, "block", units=16)
        self.assertEqual(src.count("mod g"), 3, "never more modules than trees")
        self.assertNotIn("mod g", bc.chunk_source(trees, 0, 0, self.f.n_features, self.f.groups, "block", units=1))

    def test_the_instrumented_and_the_final_build_get_the_same_shape(self):
        """A profile only matches code built the same way: the opt-level, LTO and codegen-unit flags of the profile-generate build and of the
        profile-use build are identical, in every shape."""
        seen, real = [], bc.run_measured

        def spy(cmd, env=None, watch=None):
            seen.append(list(cmd))
            return real(cmd, env, watch)

        for shape in ("fast", "top"):
            seen.clear()
            with mock.patch.object(bc, "run_measured", spy), contextlib.redirect_stderr(io.StringIO()):
                bc.CodegenPgoBackend().compile(self.f, self.tmp / "same.so", {**FAST, "shape": shape, "pgo_rows": str(self.pgo_rows)})
            shape_of = lambda c: [a for a in c if a.startswith(("opt-level=", "codegen-units=", "lto="))]  # noqa: E731
            gen = [shape_of(c) for c in seen if any(a.startswith("profile-generate=") for a in c) and "obj" in c]
            use = [shape_of(c) for c in seen if any(a.startswith("profile-use=") for a in c) and "obj" in c]
            self.assertTrue(gen and use, shape)
            self.assertEqual({tuple(g) for g in gen} | {tuple(u) for u in use}, {tuple(gen[0])}, shape)
            self.assertIn(f"codegen-units={bc.SHAPES[shape]}", gen[0])
            self.assertIn("lto=off", gen[0])
            # the driver's link is part of the shape too (`top` is one codegen unit EVERYWHERE), the same in the instrumented and the final build
            driver = lambda c: [a for a in c if a.startswith(("codegen-units=", "lto="))]  # noqa: E731
            d_gen = [driver(c) for c in seen if "cdylib" in c and any(a.startswith("profile-generate=") for a in c)]
            d_use = [driver(c) for c in seen if "cdylib" in c and any(a.startswith("profile-use=") for a in c)]
            self.assertEqual((len(d_gen), len(d_use)), (1, 1), shape)
            self.assertEqual(d_gen, d_use, shape)
            self.assertEqual(d_gen[0], ["codegen-units=1"] if shape == "top" else [], shape)  # fast: rustc's defaults, top: one unit

    def test_an_unknown_shape_is_refused_in_one_line(self):
        with self.assertRaisesRegex(ValueError, r"--shape 'turbo': give fast .* or top"):
            bc.resolve_shape({"shape": "turbo"})
        with self.assertRaises(ValueError):
            qf_compile.compile_model(FIX / "xgb_multi.ubj", self.tmp / "no.so", "codegen-block", shape="turbo")
        with self.assertRaisesRegex(ValueError, "not generated code"):
            qf_compile.compile_model(FIX / "xgb_multi.ubj", self.tmp / "no2.bin", "flat", shape="top")

    def test_the_plan_keeps_the_wanted_jobs_and_units_when_they_fit(self):
        est = bc.rustc_peak_estimate_mb
        self.assertEqual(bc.plan_build([50_000] * 4, 3, 16, 10 * est(50_000, 16) * 3), (3, 16))
        self.assertEqual(bc.plan_build([50_000] * 4, 3, 16, None), (3, 16), "no figure for the RAM: nothing to plan against")
        self.assertEqual(bc.plan_build([50_000], 3, 16, 1e9), (1, 16), "never more jobs than chunks")

    def test_the_plan_lowers_the_jobs_first_then_the_units_and_refuses_when_one_job_does_not_fit(self):
        est, share = bc.rustc_peak_estimate_mb, bc.RAM_SHARE
        chunks = [150_000] * 4
        two_jobs = 2 * est(150_000, 16) / share
        self.assertEqual(bc.plan_build(chunks, 3, 16, two_jobs + 1), (2, 16))
        one_job = est(150_000, 16) / share
        self.assertEqual(bc.plan_build(chunks, 3, 16, one_job + 1), (1, 16))
        # a little less than one job at 16 units: fewer units, one job (the estimate grows with the units, so some count below 16 fits)
        got = bc.plan_build(chunks, 3, 16, one_job - 1)
        self.assertEqual(got[0], 1)
        self.assertLess(got[1], 16)
        self.assertLessEqual(est(150_000, got[1]), (one_job - 1) * share)
        self.assertIsNone(bc.plan_build(chunks, 3, 16, est(150_000, 1) / share - 1))
        # the biggest chunks run together: a small chunk next to a big one costs what it costs
        mixed = [150_000, 1_000, 1_000]
        self.assertEqual(bc.plan_build(mixed, 2, 16, (est(150_000, 16) + est(1_000, 16)) / share + 1), (2, 16))

    def test_a_low_ram_limit_lowers_the_jobs_of_a_real_build_and_says_so_on_one_line(self):
        small = max(1, self.f.n_nodes // 3)  # three or more chunks
        per_job = bc.rustc_peak_estimate_mb(small, 16)
        info, err = self.build("ram-high", {"shape": "fast", "chunk_nodes": small, "jobs": 3, "ram_mb": 100 * per_job})
        self.assertGreaterEqual(info["chunks"], 3)
        self.assertEqual(err.strip(), "RAM: building with 3 jobs / 16 units")
        lo, hi = max(bc.RUSTC_LINK_MB, per_job), 2 * per_job  # one job fits, two do not
        self.assertLess(lo, hi)
        info2, err2 = self.build("ram-low", {"shape": "fast", "chunk_nodes": small, "jobs": 3, "ram_mb": (lo + hi) / 2 / bc.RAM_SHARE})
        self.assertEqual((info2["jobs"], info2["codegen_units"]), (1, 16))
        self.assertTrue(err2.startswith("RAM: building with 1 job / 16 units"), err2)
        self.assertEqual(len(err2.strip().splitlines()), 1)
        self.assertTrue(same_bits(*[bc.CodegenForest(self.tmp / f"{t}.so").predict(self.x, 1) for t in ("ram-high", "ram-low")]))

    def test_the_build_is_refused_not_swapped_when_one_small_job_does_not_fit(self):
        with self.assertRaises(bc.NotEnoughRam) as cm:
            self.build("no-ram", {"ram_mb": 10})
        self.assertNotIn("\n", str(cm.exception))
        self.assertIn("10 MB are free", str(cm.exception))
        self.assertIsInstance(cm.exception, ValueError)  # one line, exit 3
        self.assertFalse((self.tmp / "no-ram.so").exists())

    def test_the_ram_figure_is_the_given_one_else_the_smaller_of_memory_and_cgroup(self):
        with mock.patch.dict(bc.os.environ, {"QF_RAM_MB": "321"}):
            self.assertEqual(bc.available_ram_mb({}), 321.0)
            self.assertEqual(bc.available_ram_mb({"ram_mb": 99}), 99.0, "the option beats the environment")
        env = {k: v for k, v in bc.os.environ.items() if k != "QF_RAM_MB"}
        with mock.patch.dict(bc.os.environ, env, clear=True):
            with (
                mock.patch.object(bc, "meminfo_available_mb", return_value=5000.0),
                mock.patch.object(bc, "cgroup_headroom_mb", return_value=800.0),
            ):
                self.assertEqual(bc.available_ram_mb({}), 800.0)
            with (
                mock.patch.object(bc, "meminfo_available_mb", return_value=500.0),
                mock.patch.object(bc, "cgroup_headroom_mb", return_value=None),
            ):
                self.assertEqual(bc.available_ram_mb({}), 500.0)
            with (
                mock.patch.object(bc, "meminfo_available_mb", return_value=None),
                mock.patch.object(bc, "cgroup_headroom_mb", return_value=None),
            ):
                self.assertIsNone(bc.available_ram_mb({}))
        for bad in (0, -5, "lots"):
            with self.assertRaisesRegex(ValueError, "positive number"):
                bc.available_ram_mb({"ram_mb": bad})

    def test_the_cgroup_limit_is_read_from_the_tightest_parent(self):
        root = self.tmp / "cg"
        (root / "a" / "b").mkdir(parents=True)
        (root / "memory.max").write_text("max\n")
        (root / "a" / "memory.max").write_text("2000000000\n")
        (root / "a" / "memory.current").write_text("500000000\n")
        (root / "a" / "b" / "memory.max").write_text("max\n")
        proc = self.tmp / "proc-cgroup"
        proc.write_text("12:cpu:/x\n0::/a/b\n")
        self.assertEqual(bc.cgroup_headroom_mb(str(proc), str(root)), 1500.0)
        (root / "a" / "b" / "memory.max").write_text("1000000000\n")
        (root / "a" / "b" / "memory.current").write_text("900000000\n")
        self.assertEqual(bc.cgroup_headroom_mb(str(proc), str(root)), 100.0, "the tighter child wins")
        (root / "a" / "memory.max").write_text("max\n")
        (root / "a" / "b" / "memory.max").write_text("max\n")
        self.assertIsNone(bc.cgroup_headroom_mb(str(proc), str(root)), "no limit anywhere")
        self.assertIsNone(bc.cgroup_headroom_mb(str(self.tmp / "missing"), str(root)))


class BestBackendTest(unittest.TestCase):
    """`best`, the default of compile and optimize: the fastest build this machine can make, and a notice when something was given up."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-best-"))
        cls.model = FIX / "xgb_multi.ubj"
        cls.x = np.load(FIX / "x.npy")
        np.savez(cls.tmp / "d.npz", X=tiled(cls.x))
        np.save(cls.tmp / "d.npz.npy", tiled(cls.x))
        cls.f = forest("xgb_multi.ubj")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_choice_follows_what_the_machine_and_the_data_allow(self):
        self.assertEqual(qf_compile.best_backend(self.f, True), ("codegen-pgo", None))
        name, notice = qf_compile.best_backend(self.f, False)
        self.assertEqual(name, "codegen-block")
        self.assertIn("--data", notice)
        with mock.patch.object(bc, "llvm_profdata", side_effect=bc.ToolchainMissing("llvm-profdata not found: get it")):
            name, notice = qf_compile.best_backend(self.f, True)
            self.assertEqual(name, "codegen-block")
            self.assertIn("llvm-profdata", notice)
        with mock.patch.object(bc, "toolchain_bin", side_effect=bc.ToolchainMissing("rustc not found")):
            name, notice = qf_compile.best_backend(self.f, True)
            self.assertEqual(name, "flat")
            self.assertIn("rustc", notice)
        with mock.patch.object(bc, "too_deep", return_value=2000):
            name, notice = qf_compile.best_backend(self.f, True)
            self.assertEqual(name, "flat")
            self.assertIn("2,000 levels", notice)
        with mock.patch.object(type(self.f), "n_nodes", 7_000_000):
            name, notice = qf_compile.best_backend(self.f, True)
            self.assertEqual(name, "flat")
            self.assertIn("7,000,000 nodes", notice)

    def fake_profdata(self, folder: Path, llvm: str) -> Path:
        """An llvm-profdata of the given LLVM version whose every merge fails the way a profile of another LLVM does."""
        folder.mkdir(parents=True, exist_ok=True)
        tool = folder / "llvm-profdata"
        tool.write_text(
            f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "LLVM version {llvm}"; exit 0; fi\n'
            "echo 'warning: raw profile version mismatch: Profile uses raw profile format version = 11; expected version = 9' >&2\n"
            "echo 'error: no profile can be merged' >&2\nexit 1\n"
        )
        tool.chmod(0o755)
        return tool

    @contextlib.contextmanager
    def only_a_wrong_llvm_profdata(self, tag: str, rustc_llvm: int = 99):
        """The real rustc, but its own llvm-tools are hidden (its sysroot is an empty folder and its LLVM is `rustc_llvm`, a version nothing
        has) and a profdata of LLVM 18 is first on PATH: what CI saw on a machine whose system tools are of another LLVM than its toolchain."""
        empty = self.tmp / f"{tag}-empty"
        empty.mkdir()
        tool = self.fake_profdata(self.tmp / f"{tag}-path", "18.1.3")
        with (
            mock.patch.object(bc, "rustc_llvm", return_value=(empty, "x86_64-unknown-linux-gnu", rustc_llvm)),
            mock.patch.object(bc, "LLVM_ROOT", empty),
            mock.patch.dict(os.environ, {"PATH": f"{tool.parent}{os.pathsep}{os.environ['PATH']}"}),
        ):
            yield tool

    @contextlib.contextmanager
    def ci_like_layout(self, tag: str, wrong_first: Path | None = None):
        """No RUST_DIR, rustc found on PATH in a proxy folder (a cargo home's bin) that runs the real toolchain's rustc and holds no lib/rustlib
        of its own, and optionally an llvm-profdata of another LLVM first on PATH: the layout of the first CI run, which this machine's
        RUST_DIR hides."""
        real = bc.toolchain_bin() / "rustc"
        proxy = self.tmp / f"{tag}-cargo-home" / "bin"
        proxy.mkdir(parents=True)
        (proxy / "rustc").write_text(f'#!/bin/sh\nexec "{real}" "$@"\n')
        (proxy / "rustc").chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k != "RUST_DIR"}
        env["PATH"] = os.pathsep.join([*([str(wrong_first)] if wrong_first else []), str(proxy), env["PATH"]])
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(bc.toolchain_bin(), proxy)
            yield proxy

    def test_a_wrong_version_llvm_profdata_first_on_path_does_not_break_the_profile_guided_build(self):
        """The CI failure of 2026-10-23: the toolchain's own llvm-profdata must win over one on PATH whose profiles it cannot read, with rustc
        found in rustup's proxy folder."""
        tool = self.fake_profdata(self.tmp / "wrong-first", "18.1.3")
        with self.ci_like_layout("wrong-first", tool.parent):
            self.assertNotEqual(bc.llvm_profdata(), tool)
            info = qf_compile.compile_model(self.model, self.tmp / "wrong-first.so", data=self.tmp / "d.npz")
        self.assertEqual(info["backend"], "codegen-pgo")
        self.assertNotIn("notice", info)

    def test_without_a_matching_llvm_profdata_best_builds_codegen_block_and_the_notice_names_both_versions(self):
        with self.only_a_wrong_llvm_profdata("best-notice") as tool:
            name, notice = qf_compile.best_backend(self.f, True)
            self.assertEqual(name, "codegen-block")
            self.assertEqual(len(notice.splitlines()), 1)
            for word in ("LLVM 99", "LLVM 18", str(tool), "codegen-block", "rustup component add llvm-tools", "apt install llvm"):
                self.assertIn(word, notice)
            info = qf_compile.compile_model(self.model, self.tmp / "no-match.so", data=self.tmp / "d.npz")
        self.assertEqual(info["backend"], "codegen-block")
        self.assertEqual(info["notice"], notice)  # the one line the choice gave, unchanged

    def test_the_profile_guided_backend_asked_for_by_name_without_a_matching_llvm_profdata_refuses_first_in_one_line(self):
        out = self.tmp / "named-no-match.so"
        with (
            self.only_a_wrong_llvm_profdata("named"),
            mock.patch.object(bc.CodegenBackend, "_compile_in", side_effect=AssertionError("a build was started")),
            self.assertRaises(bc.ProfdataMissing) as cm,
        ):
            qf_compile.compile_model(self.model, out, backend="codegen-pgo", data=self.tmp / "d.npz")
        self.assertEqual(len(str(cm.exception).splitlines()), 1)
        for word in ("LLVM 99", "LLVM 18"):
            self.assertIn(word, str(cm.exception))
        self.assertFalse(out.exists())

    def test_the_command_line_refuses_the_profile_guided_backend_without_a_matching_llvm_profdata_in_one_line_and_exit_3(self):
        err = io.StringIO()
        with (
            self.only_a_wrong_llvm_profdata("cli"),
            mock.patch.object(
                sys,
                "argv",
                [
                    "compile.py",
                    str(self.model),
                    str(self.tmp / "cli-pgo.so"),
                    "--backend",
                    "codegen-pgo",
                    "--data",
                    str(self.tmp / "d.npz"),
                ],
            ),
            contextlib.redirect_stderr(err),
            self.assertRaises(SystemExit) as cm,
        ):
            qf_compile.main()
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())
        self.assertIn("LLVM 99", err.getvalue())
        self.assertFalse((self.tmp / "cli-pgo.so").exists())

    def test_optimize_without_a_matching_llvm_profdata_builds_codegen_block_and_asked_for_by_name_reports_not_compiled(self):
        with self.only_a_wrong_llvm_profdata("opt"):
            rep = optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-no-match", "100%", True, True)
            self.assertTrue(rep["ok"], rep["failures"])
            self.assertIn("compile", rep["stages"], rep["compile_unsupported"])
            self.assertEqual(rep["stages"]["compile"]["backend"], "codegen-block")
            self.assertIn("LLVM 18", rep["stages"]["compile"]["notice"])
            self.assertTrue(rep["stages"]["verify"]["ok"])
            out = self.tmp / "opt-named"
            rep = optimize.optimize(self.model, self.tmp / "d.npz", out, "100%", True, True, backend="codegen-pgo")
        self.assertTrue(rep["ok"], rep["failures"])  # not a failed bar: the pruned models stay
        self.assertEqual(len(rep["compile_unsupported"].splitlines()), 1)
        self.assertIn("LLVM 99", rep["compile_unsupported"])
        self.assertNotIn("compile", rep["stages"])  # no half result: no stage, no verify, no artifact
        self.assertNotIn("verify", rep["stages"])
        self.assertEqual(sorted(p.name for p in out.glob("model.*")), [])
        self.assertIn("NOT COMPILED", (out / "report.md").read_text())

    def test_a_profile_of_too_few_rows_makes_best_build_codegen_block_and_a_named_backend_fail(self):
        few = self.tmp / "few.npz"
        np.savez(few, X=self.x[:10])
        info = qf_compile.compile_model(self.model, self.tmp / "few.so", data=few)
        self.assertEqual(info["backend"], "codegen-block")
        self.assertEqual(len(info["notice"].splitlines()), 1)
        for word in ("10 profile rows are too few", f"{bc.PROFILE_MIN_ROWS:,}", "built codegen-block", "--data"):
            self.assertIn(word, info["notice"])
        self.assertNotIn("profile_rows", info)
        out = self.tmp / "few-named.so"
        with (
            mock.patch.object(bc.CodegenBackend, "_compile_in", side_effect=AssertionError("a build was started")),
            self.assertRaises(bc.ProfileUnreliable) as cm,
        ):
            qf_compile.compile_model(self.model, out, backend="codegen-pgo", data=few)
        self.assertIsInstance(cm.exception, ValueError)  # every command prints it as one line and exits 3
        self.assertEqual(len(str(cm.exception).splitlines()), 1)
        self.assertFalse(out.exists())

    def test_a_profile_whose_halves_disagree_makes_best_build_codegen_block_and_a_named_backend_fail(self):
        with mock.patch.object(bc, "PROFILE_MIN_OVERLAP", 100.1):  # no two halves can agree on more than all of it
            info = qf_compile.compile_model(self.model, self.tmp / "halves.so", data=self.tmp / "d.npz")
            self.assertEqual(info["backend"], "codegen-block")
            self.assertEqual(len(info["notice"].splitlines()), 1)
            for word in ("two halves of the", "profile rows give profiles that agree on", "edge counts", "built codegen-block"):
                self.assertIn(word, info["notice"])
            out = self.tmp / "halves-named.so"
            with self.assertRaisesRegex(bc.ProfileUnreliable, "agree on"):
                qf_compile.compile_model(self.model, out, backend="codegen-pgo", data=self.tmp / "d.npz")
            self.assertFalse(out.exists())
            rep = optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-halves", "100%", True, True)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertIn("compile", rep["stages"], rep["compile_unsupported"])
        self.assertEqual(rep["stages"]["compile"]["backend"], "codegen-block")
        self.assertIn("agree on", rep["stages"]["compile"]["notice"])
        self.assertTrue(rep["stages"]["verify"]["ok"])

    def test_a_good_profile_is_reported_with_its_rows_and_the_agreement_of_its_halves(self):
        info = qf_compile.compile_model(self.model, self.tmp / "reported.so", data=self.tmp / "d.npz")
        self.assertEqual(info["backend"], "codegen-pgo")
        self.assertEqual(info["profile_rows"], len(np.load(self.tmp / "d.npz")["X"]))
        self.assertGreaterEqual(info["profile_overlap"], bc.PROFILE_MIN_OVERLAP)
        self.assertGreaterEqual(info["profile_function_overlap"], 90.0)  # reported, not gated: every tree function runs once per block
        self.assertLessEqual(info["profile_overlap"], 100.0)
        rep = optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-reported", "100%", True, True)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertIn("compile", rep["stages"], rep["compile_unsupported"])
        self.assertEqual(rep["stages"]["compile"]["profile_rows"], info["profile_rows"])
        self.assertRegex(
            (self.tmp / "opt-reported" / "report.md").read_text(),
            r"- profile: [\d,]+ rows; two halves of them agree on [\d.]+ % of the edge",
        )

    def test_the_profile_is_two_disjoint_halves_of_the_rows_and_a_coverage_pass(self):
        seen = {}
        real = bc.profile_run

        def spy(inst, rows, raw, kind):
            seen[raw.name.split("-")[1]] = (np.load(rows), kind)
            return real(inst, rows, raw, kind)

        with mock.patch.object(bc, "profile_run", spy):
            info = bc.CodegenPgoBackend().compile(self.f, self.tmp / "halves-spy.so", {**FAST, "pgo_rows": str(self.tmp / "d.npz.npy")})
        allrows = np.load(self.tmp / "d.npz.npy")
        order = lambda a: a[np.lexsort(a.T[::-1])]  # noqa: E731
        self.assertEqual((seen["a"][1], seen["b"][1], seen["c"][1]), ("main", "main", "cover"))
        halves = np.concatenate([seen["a"][0], seen["b"][0]])
        np.testing.assert_array_equal(order(halves), order(allrows))  # together all of the rows, once each
        self.assertEqual(len(seen["a"][0]) + len(seen["b"][0]), len(allrows))
        self.assertLessEqual(abs(len(seen["a"][0]) - len(seen["b"][0])), 1)
        self.assertEqual(len(seen["c"][0]), int(bc.PROFILE_COVER_FACTOR * bc.CodegenPgoBackend.default_block))
        self.assertEqual(info["profile_rows"], len(allrows))

    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0, "root can write anywhere")
    def test_an_unwritable_output_folder_is_refused_first_with_one_line_and_nothing_is_built(self):
        ro = self.tmp / "readonly"
        ro.mkdir()
        ro.chmod(0o555)
        try:
            for backend in ("best", "codegen-pgo", "codegen-block", "flat", "auto"):
                err = io.StringIO()
                with (
                    contextlib.redirect_stderr(err),
                    mock.patch.object(qf, "load_forest", side_effect=AssertionError("read the model")),
                    self.assertRaisesRegex(ValueError, "the folder is not writable") as cm,
                ):
                    qf_compile.compile_model(self.model, ro / "m.so", backend, data=self.tmp / "d.npz")
                self.assertEqual(err.getvalue(), "", backend)  # no "RAM: building ..." line for a build that never starts
                self.assertEqual(len(str(cm.exception).splitlines()), 1)
            self.assertEqual(list(ro.iterdir()), [])
        finally:
            ro.chmod(0o755)

    def test_too_little_free_ram_falls_back_to_the_flat_format_with_a_notice(self):
        with mock.patch.dict(bc.os.environ, {"QF_RAM_MB": "20"}):
            name, notice = qf_compile.best_backend(self.f, True)
        self.assertEqual(name, "flat")
        self.assertIn("20 MB of RAM are free", notice)
        self.assertIn("without swapping", notice)

    def test_compile_with_data_is_profile_guided_and_loads_by_what_was_built(self):
        out = self.tmp / "with-data.bin"  # a suffix that says nothing
        info = qf_compile.compile_model(self.model, out, data=self.tmp / "d.npz")
        self.assertEqual((info["backend"], info["target_cpu"]), ("codegen-pgo", bc.default_cpu()))
        self.assertNotIn("notice", info)
        self.assertEqual(json.loads(Path(f"{out}.choice.json").read_text()), {"backend": "codegen-pgo"})
        flat = self.tmp / "ref.flat"
        qf_compile.compile_model(self.model, flat, backend="flat")
        self.assertTrue(same_bits(backends.load_artifact(out).predict(self.x, 1), backends.load_artifact(flat).predict(self.x, 1)))

    def test_compile_without_data_is_codegen_block_with_one_notice(self):
        out = self.tmp / "no-data.so"
        info = qf_compile.compile_model(self.model, out)
        self.assertEqual(info["backend"], "codegen-block")
        self.assertIn("--data", info["notice"])
        self.assertNotIn("\n", info["notice"])

    def test_a_name_that_says_the_other_kind_of_file_gets_one_line_naming_what_it_holds(self):
        """`compile m.ubj m.flat` with the default builds generated code (the loader is named in m.flat.choice.json): the notice says what the
        file holds and how to get the flat format; the same for a `.so` when only the flat format could be built. A name that matches the
        build, or a backend asked for by name, gets no such line."""
        out = self.tmp / "named.flat"
        info = qf_compile.compile_model(self.model, out)
        self.assertEqual(info["backend"], "codegen-block")
        self.assertIn("named.flat holds generated code, not the flat format", info["notice"])
        self.assertIn("--backend flat", info["notice"])
        self.assertIn("--data", info["notice"])  # the notice of the build itself (no profile rows) stays, on the same line
        self.assertEqual(len(info["notice"].splitlines()), 1)
        self.assertEqual(backends.load_artifact(out).predict(self.x, 1).shape[0], len(self.x))
        with mock.patch.object(bc, "toolchain_bin", side_effect=bc.ToolchainMissing("rustc not found")):
            info = qf_compile.compile_model(self.model, self.tmp / "fallback.so")
        self.assertEqual(info["backend"], "flat")
        self.assertIn("rustc not found", info["notice"])
        self.assertIn("fallback.so holds the flat format, not the generated code", info["notice"])
        self.assertEqual(len(info["notice"].splitlines()), 1)
        self.assertNotIn("notice", qf_compile.compile_model(self.model, self.tmp / "match.flat", backend="flat"))
        self.assertNotIn("notice", qf_compile.compile_model(self.model, self.tmp / "match-pgo.so", data=self.tmp / "d.npz"))
        self.assertNotIn("notice", qf_compile.compile_model(self.model, self.tmp / "asked.flat", backend="codegen-block"))

    def test_the_profile_guided_backend_asked_for_by_name_still_needs_its_data(self):
        with self.assertRaisesRegex(ValueError, "give --data"):
            qf_compile.compile_model(self.model, self.tmp / "named.so", backend="codegen-pgo")

    def test_a_cpu_cannot_be_chosen_for_a_backend_that_is_not_generated_code(self):
        with self.assertRaisesRegex(ValueError, "not generated code"):
            qf_compile.compile_model(self.model, self.tmp / "x.flat", backend="flat", target_cpu="native")

    def test_the_command_line_refuses_a_cpu_in_one_line_and_exit_3(self):
        for argv, word in (
            (["--target-cpu", "nosuchcpu-v9"], "does not know"),
            (["--native", "--target-cpu", "x86-64-v3"], "alternatives"),
        ):
            err = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["compile.py", str(self.model), str(self.tmp / "cli.so"), *argv]),
                contextlib.redirect_stderr(err),
                self.assertRaises(SystemExit) as cm,
            ):
                qf_compile.main()
            self.assertEqual(cm.exception.code, 3)
            self.assertIn(word, err.getvalue())
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1)

    def test_optimize_compiles_the_best_build_and_the_report_names_it(self):
        rep = optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-best", "100%", True, True)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertIn("compile", rep["stages"], rep["compile_unsupported"])  # a build that failed says why here, not as a KeyError
        c = rep["stages"]["compile"]
        self.assertEqual((c["backend"], c["target_cpu"]), ("codegen-pgo", bc.default_cpu()))
        self.assertTrue(rep["stages"]["verify"]["ok"])
        self.assertTrue((self.tmp / "opt-best" / "model.so").is_file())
        self.assertIn(
            f"compile (codegen-pgo, built for {bc.default_cpu()} CPU, {bc.DEFAULT_SHAPE} shape)",
            (self.tmp / "opt-best" / "report.md").read_text(),
        )

    def test_optimize_native_is_an_opt_in_and_the_report_says_where_it_runs(self):
        rep = optimize.optimize(
            self.model, self.tmp / "d.npz", self.tmp / "opt-native", "100%", True, True, backend="codegen-block", target_cpu="native"
        )
        self.assertTrue(rep["ok"], rep["failures"])  # --verify loaded it: this machine is the CPU type it was built for
        self.assertIn("compile", rep["stages"], rep["compile_unsupported"])
        self.assertEqual(rep["stages"]["compile"]["target_cpu"], "native")
        self.assertIn("built for native CPU (runs only on this CPU type)", (self.tmp / "opt-native" / "report.md").read_text())

    def test_optimize_verify_of_a_build_for_a_cpu_this_one_lacks_keeps_pruned_and_reports_it_not_compiled(self):
        out = self.tmp / "opt-foreign"
        with mock.patch.object(bc, "CPUINFO", cpuinfo(self.tmp / "nothing.txt", set())):  # a build is checked when it is loaded
            rep = optimize.optimize(self.model, self.tmp / "d.npz", out, "100%", True, True, backend="codegen-block")
        self.assertTrue(rep["ok"], rep["failures"])  # not a failed bar
        self.assertIn("illegal instruction", rep["compile_unsupported"])
        self.assertNotIn("compile", rep["stages"])
        self.assertNotIn("verify", rep["stages"])
        self.assertTrue((out / "pruned").is_dir())
        self.assertEqual(sorted(p.name for p in out.glob("model.*")), [])
        self.assertIn("NOT COMPILED", (out / "report.md").read_text())

    def test_optimize_without_rustc_falls_back_to_flat_with_a_notice(self):
        with mock.patch.object(bc, "toolchain_bin", side_effect=bc.ToolchainMissing("rustc not found")):
            rep = optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-norustc", "100%", True, True)
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertIn("compile", rep["stages"], rep["compile_unsupported"])
        self.assertEqual(rep["stages"]["compile"]["backend"], "flat")
        self.assertIn("rustc", rep["stages"]["compile"]["notice"])
        self.assertTrue((self.tmp / "opt-norustc" / "model.flat").is_file())
        self.assertIn("- note: rustc not found", (self.tmp / "opt-norustc" / "report.md").read_text())

    def test_a_cpu_asked_for_without_compile_is_refused_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "add --compile"):
            optimize.optimize(self.model, self.tmp / "d.npz", self.tmp / "opt-nocompile", "100%", target_cpu="native")
        self.assertFalse((self.tmp / "opt-nocompile").exists())


if __name__ == "__main__":
    unittest.main()
