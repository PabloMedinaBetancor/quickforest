"""Where `codegen-pgo` finds `llvm-profdata`: the one of rustc's own LLVM, first the rustup llvm-tools copy in the sysroot that `rustc --print sysroot`
names (never a guess from the folder rustc was found in: with rustc on PATH that is rustup's proxy folder), then a PATH or /usr/lib/llvm-*/bin
one only when it reports rustc's LLVM major version; never another version, and the message when there is none. Fake executables in
temporary folders; needs numpy only.

    python3 -s -m unittest test_profdata
"""

import contextlib
import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import backend_codegen

HOST = "x86_64-unknown-linux-gnu"


def fake_tool(folder: Path, llvm: str | None = "18.1.3", body: str = "exit 0") -> Path:
    """An llvm-profdata that answers `--version` the way the real one does (nothing useful when `llvm` is None), else runs `body`."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "llvm-profdata"
    said = f"echo 'LLVM (http://llvm.org/):'; echo '  LLVM version {llvm}'; echo '  Optimized build.'" if llvm else "echo 'no idea'"
    p.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then {said}; exit 0; fi\n{body}\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def fake_rustc(folder: Path, sysroot: Path, llvm: str | None, host: str | None = HOST) -> Path:
    """A rustc that prints `-vV` and `--print sysroot` the way the real one does (no LLVM line when `llvm` is None, no host line when None)."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "rustc"
    lines = ["echo 'rustc 1.99.0 (abcdef123 2026-01-01)'"]
    lines += [f"echo 'host: {host}'"] if host else []
    lines += [f"echo 'LLVM version: {llvm}'"] if llvm else []
    p.write_text(
        f'#!/bin/sh\nif [ "$1" = "--print" ] && [ "$2" = "sysroot" ]; then echo {sysroot}; exit 0; fi\n'
        f'if [ "$1" = "-vV" ]; then {"; ".join(lines)}; exit 0; fi\nexit 1\n'
    )
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


class LlvmProfdataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self._enter(tempfile.TemporaryDirectory()))
        self.sysroot = self.tmp / "rustup" / "toolchains" / f"1.99.0-{HOST}"  # where `rustc --print sysroot` says the toolchain is
        self.rustlib = self.sysroot / "lib" / "rustlib" / HOST / "bin"
        self.bin = self.tmp / "cargo-home" / "bin"  # where rustc is found: rustup's proxy folder, which holds no lib/rustlib
        self.path = self.tmp / "path"  # the only folder on PATH: no real llvm-profdata can leak in
        self.path.mkdir()
        self.llvm_root = self.tmp / "usr-lib"
        self.llvm_root.mkdir()
        for p in (
            mock.patch.object(backend_codegen, "toolchain_bin", lambda: self.bin),
            mock.patch.object(backend_codegen, "LLVM_ROOT", self.llvm_root),
            mock.patch.dict(os.environ, {"PATH": str(self.path)}),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _enter(self, cm):
        self.addCleanup(cm.__exit__, None, None, None)
        return cm.__enter__()

    def rustc(self, llvm: str | None = "23.1.0", host: str | None = HOST) -> None:
        fake_rustc(self.bin, self.sysroot, llvm, host)

    def pick(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            return backend_codegen.llvm_profdata(), err.getvalue()

    def refused(self) -> str:
        with self.assertRaises(backend_codegen.ProfdataMissing) as cm:
            backend_codegen.llvm_profdata()
        msg = str(cm.exception)
        self.assertEqual(len(msg.splitlines()), 1, msg)
        self.assertIn("rustup component add llvm-tools", msg)
        self.assertIn("apt install llvm", msg)
        self.assertIn("same LLVM major version as rustc", msg)
        self.assertIn("rustc -vV", msg)
        return msg

    def test_the_sysroots_rustlib_comes_first_whatever_else_is_installed(self):
        self.rustc("23.1.0")
        want = fake_tool(self.rustlib, "23.1.0")
        fake_tool(self.path, "23.1.0")
        fake_tool(self.llvm_root / "llvm-23" / "bin", "23.1.0")
        self.assertEqual(self.pick(), (want, ""))

    def test_a_proxy_folder_on_path_does_not_hide_the_toolchains_llvm_tools(self):
        """The CI failure of 2026-10-23: rustc was found in rustup's proxy folder (cargo home/bin), whose parent holds no lib/rustlib, so the
        toolchain's own llvm-profdata was missed and a wrong-version one on PATH was used."""
        self.rustc("23.1.0")
        self.assertFalse((self.bin.parent / "lib").exists())
        want = fake_tool(self.rustlib, "23.1.0")
        fake_tool(self.path, "18.1.3", "echo 'error: raw profile version mismatch'; exit 1")  # first on PATH, of another LLVM
        fake_tool(self.llvm_root / "llvm-18" / "bin", "18.1.3")
        self.assertEqual(self.pick(), (want, ""))

    def test_the_sysroot_tool_is_found_when_rustc_prints_no_host_line(self):
        self.rustc("23.1.0", host=None)
        want = fake_tool(self.rustlib, "23.1.0")
        self.assertEqual(self.pick(), (want, ""))

    def test_then_path_when_it_is_of_rustcs_llvm(self):
        self.rustc("18.1.3")
        want = fake_tool(self.path, "18.1.3")
        fake_tool(self.llvm_root / "llvm-21" / "bin", "21.1.0")
        self.assertEqual(self.pick(), (want, ""))

    def test_path_of_another_llvm_is_never_used_and_the_message_names_both_versions(self):
        self.rustc("23.1.0")
        old = fake_tool(self.path, "18.1.3")
        msg = self.refused()
        for word in ("LLVM 23", "LLVM 18", str(old)):
            self.assertIn(word, msg)

    def test_then_the_usr_lib_llvm_of_rustcs_version_not_the_highest(self):
        self.rustc("18.1.3")
        for v in ("9", "18", "21"):
            fake_tool(self.llvm_root / f"llvm-{v}" / "bin", f"{v}.1.0")
        want = self.llvm_root / "llvm-18" / "bin" / "llvm-profdata"
        self.assertEqual(self.pick(), (want, ""))

    def test_usr_lib_llvm_of_other_versions_only_is_refused_naming_the_versions_found(self):
        self.rustc("20.1.5")
        for v in ("18", "21"):
            fake_tool(self.llvm_root / f"llvm-{v}" / "bin", f"{v}.1.0")
        msg = self.refused()
        for word in ("LLVM 20", "LLVM 18", "LLVM 21"):
            self.assertIn(word, msg)

    def test_the_folder_name_is_not_trusted_for_the_version(self):
        self.rustc("18.1.3")
        fake_tool(self.llvm_root / "llvm-18" / "bin", "21.1.0")  # a folder called llvm-18 holding another version
        self.assertIn("LLVM 21", self.refused())

    def test_a_tool_whose_version_cannot_be_read_is_not_used(self):
        self.rustc("18.1.3")
        fake_tool(self.path, None)
        self.assertIn("unreadable version", self.refused())

    def test_an_unreadable_rustc_version_uses_only_the_sysroot_tool(self):
        self.rustc(None)
        fake_tool(self.path, "18.1.3")
        fake_tool(self.llvm_root / "llvm-18" / "bin", "18.1.3")
        self.assertIn("could not be read", self.refused())
        want = fake_tool(self.rustlib, "23.1.0")
        self.assertEqual(self.pick(), (want, ""))

    def test_a_rustc_that_does_not_run_finds_nothing_by_guessing(self):
        self.bin.mkdir(parents=True)  # no rustc in it
        fake_tool(self.bin.parent / "lib" / "rustlib" / HOST / "bin", "23.1.0")  # where the old guess looked
        fake_tool(self.path, "18.1.3")
        self.assertIn("could not be read", self.refused())

    def test_none_says_both_ways_out_and_the_version_rule(self):
        self.rustc("23.1.0")
        self.assertIn("LLVM 23", self.refused())

    def test_the_refusal_is_a_toolchain_refusal_a_caller_can_catch_as_before(self):
        self.rustc("23.1.0")
        with self.assertRaises(FileNotFoundError) as cm:  # choose.can_run and the unit tests that skip catch it as this
            backend_codegen.llvm_profdata()
        self.assertIsInstance(cm.exception, backend_codegen.ToolchainMissing)

    def test_a_rejected_profile_shows_the_tools_message_and_the_hint(self):
        self.rustc("18.1.3")
        fake_tool(self.path, "18.1.3", "echo 'error: unsupported instrumentation profile format version'; exit 1")
        with self.assertRaises(RuntimeError) as cm:
            backend_codegen.merge_profiles(self.tmp / "m.profdata", [self.tmp / "a.profraw"])
        msg = str(cm.exception)
        self.assertIn("unsupported instrumentation profile format version", msg)
        self.assertIn("same LLVM major version as rustc", msg)

    def test_a_good_merge_returns_quietly(self):
        self.rustc("18.1.3")
        fake_tool(self.path, "18.1.3", "exit 0")
        backend_codegen.merge_profiles(self.tmp / "m.profdata", [self.tmp / "a.profraw"])

    def test_the_edge_overlap_is_read_from_the_tools_program_level_line(self):
        self.rustc("18.1.3")
        fake_tool(self.path, "18.1.3", "echo 'Program level:'; echo '  # of functions overlap: 88'; echo '  Edge profile overlap: 97.250%'")
        self.assertEqual(backend_codegen.edge_overlap(self.tmp / "a.profdata", self.tmp / "b.profdata"), 97.25)

    def test_an_overlap_nobody_can_read_fails_the_build_in_one_line(self):
        self.rustc("18.1.3")
        for body in ("echo 'Program level:'", "echo 'Edge profile overlap: 99.000%'; exit 1"):
            fake_tool(self.path, "18.1.3", body)
            with self.assertRaisesRegex(backend_codegen.BuildFailed, "no readable `Edge profile overlap` line") as cm:
                backend_codegen.edge_overlap(self.tmp / "a.profdata", self.tmp / "b.profdata")
            self.assertEqual(len(str(cm.exception).splitlines()), 1)


class FunctionOverlapTest(unittest.TestCase):
    def test_it_is_the_sum_of_the_smaller_share_of_the_counts_of_each_function(self):
        same = {"f": ("h", 70), "g": ("h", 30)}
        self.assertEqual(backend_codegen.function_overlap(same, dict(same)), 100.0)
        self.assertEqual(backend_codegen.function_overlap({"f": ("h", 50), "g": ("h", 50)}, {"f": ("h", 100)}), 50.0)
        self.assertEqual(backend_codegen.function_overlap({"f": ("h", 1)}, {"g": ("h", 1)}), 0.0)
        self.assertAlmostEqual(backend_codegen.function_overlap({"f": ("h", 60), "g": ("h", 40)}, {"f": ("h", 30), "g": ("h", 70)}), 70.0)

    def test_a_profile_without_counts_overlaps_nothing(self):
        self.assertEqual(backend_codegen.function_overlap({"f": ("h", 0)}, {"f": ("h", 5)}), 0.0)
        self.assertEqual(backend_codegen.function_overlap({}, {}), 0.0)


class ProxyLayoutTest(unittest.TestCase):
    """The layout CI has: no RUST_DIR, rustc found on PATH in a rustup-like proxy folder (cargo home/bin) that runs the real toolchain's rustc
    and holds no lib/rustlib of its own. The real rustc and its llvm-tools are used; skipped where there are none."""

    def test_rustc_found_through_a_proxy_folder_still_finds_the_toolchains_llvm_profdata(self):
        try:
            real = backend_codegen.toolchain_bin() / "rustc"
            sysroot, host, _ = backend_codegen.rustc_llvm()
        except backend_codegen.ToolchainMissing as e:
            self.skipTest(str(e))
        want = sysroot / "lib" / "rustlib" / host / "bin" / "llvm-profdata" if sysroot and host else None
        if want is None or not want.is_file():
            self.skipTest("this toolchain has no llvm-tools component")
        with tempfile.TemporaryDirectory() as tmp:
            proxy = Path(tmp, "cargo-home", "bin")
            proxy.mkdir(parents=True)
            (proxy / "rustc").write_text(f'#!/bin/sh\nexec "{real}" "$@"\n')
            (proxy / "rustc").chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != "RUST_DIR"}
            env["PATH"] = f"{proxy}{os.pathsep}{env['PATH']}"
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(backend_codegen.toolchain_bin(), proxy)
                self.assertFalse((proxy.parent / "lib").exists())  # the old guess (the proxy's parent) holds nothing
                self.assertEqual(backend_codegen.llvm_profdata(), want)


class NoRustcTest(unittest.TestCase):
    """`quickforest compile --backend codegen` on a machine with no rustc ends in a one-line refusal, exit 3, not a traceback."""

    def test_compile_without_rustc_is_a_one_line_refusal_with_exit_3(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "lgb_plain.txt"
        env = {k: v for k, v in os.environ.items() if k not in ("RUST_DIR", "PATH")}
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp, "empty-path")
            empty.mkdir()
            env["PATH"] = str(empty)  # nothing on PATH: no rustc, and RUST_DIR is gone too
            p = subprocess.run(
                [sys.executable, "-I", str(SCRIPTS / "compile.py"), str(fixture), str(Path(tmp, "m.so")), "--backend", "codegen"],
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        lines = p.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1, p.stderr)
        for word in ("rustc not found", "RUST_DIR", "PATH", "rustup", "apt install"):
            self.assertIn(word, lines[0])


if __name__ == "__main__":
    unittest.main()
