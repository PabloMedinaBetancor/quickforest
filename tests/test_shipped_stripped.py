"""Every library QuickForest ships or leaves for the user is STRIPPED (the libraries it compiles for the user too): no `.symtab` section (the symbol
table with the function names), i.e. what `file` reports as "stripped". The exported dynamic table (`.dynsym`, the entry points the loader needs) stays.

Checked: the shipped evaluator libraries under bin/, and everything `compile` leaves in the output folder for the plain chunks build, the one-crate build, the
profile-guided build and the CS-PGO build (REAL rustc on the toy forest; the PGO ones need llvm-profdata of rustc's LLVM and skip without it; the CS-PGO one
also needs lld). The instrumented generation builds may keep names, they never reach the output; this test looks at the output folder, so it proves that.

    python3 -s -m unittest test_shipped_stripped
"""

import contextlib
import io
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as bc
from test_compile_settings import FIX, forest
from tiled import tiled

LIB_SUFFIXES = (".so", ".dll", ".dylib")
SHT_SYMTAB = 2


def has_symbol_table(path: Path) -> bool:
    """True when an ELF file carries a `.symtab` section (SHT_SYMTAB), or a PE/Mach-O file is not provably stripped (those are refused as unknown)."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        raise AssertionError(f"{path.name}: not an ELF file (a Windows .dll / Mach-O must be checked on its own platform)")
    is64, little = data[4] == 2, data[5] == 1
    e = "<" if little else ">"
    if is64:
        shoff, shentsize, shnum = struct.unpack_from(e + "Q", data, 0x28)[0], *struct.unpack_from(e + "HH", data, 0x3A)
        types = [struct.unpack_from(e + "I", data, shoff + i * shentsize + 4)[0] for i in range(shnum)]
    else:
        shoff, shentsize, shnum = struct.unpack_from(e + "I", data, 0x20)[0], *struct.unpack_from(e + "HH", data, 0x2E)
        types = [struct.unpack_from(e + "I", data, shoff + i * shentsize + 4)[0] for i in range(shnum)]
    return SHT_SYMTAB in types


def libs_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in LIB_SUFFIXES)


class CheckerTest(unittest.TestCase):
    def test_the_checker_tells_a_stripped_library_from_one_with_names(self):
        import subprocess

        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("no C compiler to build the two probe libraries")
        tmp = Path(tempfile.mkdtemp(prefix="quickforest-strip-probe-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "a.c").write_text("int f(int x){return x+1;}\n")
        subprocess.run([cc, "-shared", "-fPIC", "-o", str(tmp / "named.so"), str(tmp / "a.c")], check=True)
        subprocess.run([cc, "-shared", "-fPIC", "-s", "-o", str(tmp / "stripped.so"), str(tmp / "a.c")], check=True)
        self.assertTrue(has_symbol_table(tmp / "named.so"))
        self.assertFalse(has_symbol_table(tmp / "stripped.so"))


class ShippedLibrariesTest(unittest.TestCase):
    def test_every_library_under_bin_is_stripped(self):
        libs = libs_in(HERE.parent / "bin")
        self.assertTrue(libs, "no shipped library under bin/")
        for p in libs:
            if p.suffix == ".so":
                self.assertFalse(has_symbol_table(p), f"{p.name} keeps a symbol table")


@unittest.skipUnless(shutil.which("rustc") or os.environ.get("RUST_DIR"), "needs rustc")
class CompiledLibrariesTest(unittest.TestCase):
    """What `compile` leaves for the user, built with the real toolchain on the toy forest."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.rustc = str(bc.toolchain_bin() / "rustc")
        except bc.ToolchainMissing as e:
            raise unittest.SkipTest(str(e)) from None
        cls.tmp = Path(tempfile.mkdtemp(prefix="quickforest-strip-"))
        cls.f, cls.x = forest(), np.load(FIX / "x.npy")
        cls.rows = cls.tmp / "rows.npy"
        np.save(cls.rows, tiled(cls.x))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, tag, backend, **opts):
        out = self.tmp / tag
        out.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            info = backend().compile(self.f, out / "m.so", {"jobs": 1, "chunk_nodes": 60, "pgo_rows": str(self.rows), **opts})
        return out, info

    def assert_stripped(self, out: Path):
        libs = libs_in(out)
        self.assertTrue(libs, f"no library left in {out.name}")
        for p in libs:
            self.assertFalse(has_symbol_table(p), f"{out.name}/{p.name} keeps a symbol table")

    def need_profdata(self):
        try:
            bc.llvm_profdata()
        except Exception as e:
            self.skipTest(str(e))

    def test_codegen_block_chunks(self):
        out, info = self.build("block-chunks", bc.CodegenBlockBackend, unit="chunks")
        self.assertEqual(info["unit"], "chunks")
        self.assert_stripped(out)

    def test_codegen_block_single(self):
        out, info = self.build("block-single", bc.CodegenBlockBackend, unit="single")
        self.assertEqual(info["unit"], "single")
        self.assert_stripped(out)

    def test_codegen_pgo_chunks_without_cspgo(self):
        self.need_profdata()
        out, info = self.build("pgo-chunks", bc.CodegenPgoBackend, unit="chunks", cspgo="off")
        self.assertFalse(info["cspgo"])
        self.assert_stripped(out)

    def test_codegen_pgo_single(self):
        self.need_profdata()
        out, info = self.build("pgo-single", bc.CodegenPgoBackend, unit="single")
        self.assertEqual(info["unit"], "single")
        self.assert_stripped(out)

    def test_codegen_cspgo(self):
        self.need_profdata()
        why = bc.cspgo_unavailable(bc.resolve_compile({}), "chunks", self.rustc)
        if why:
            self.skipTest(why)
        out, info = self.build("cspgo", bc.CodegenPgoBackend, unit="chunks", cspgo="on")
        self.assertTrue(info["cspgo"])
        self.assert_stripped(out)


if __name__ == "__main__":
    unittest.main()
