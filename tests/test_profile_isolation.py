"""`profile_run` starts python with `-I`, which hides a numpy installed in the user site («No module named numpy»): QF 1.0.1 passes the parent's site folders.

Light, no rustc:   python3 -m unittest test_profile_isolation   (not `-s`: this machine's numpy lives in the user site)
"""

import json
import os
import re
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
import backend_codegen as bc


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
        raw = Path(tempfile.mkdtemp(prefix="qf-raw-")) / "raw"
        self.addCleanup(shutil.rmtree, raw.parent, True)
        script = bc.PROFILE_SCRIPT.split("from backend_codegen")[0] + "print('NUMPY-OK')\n"
        with (
            mock.patch.object(bc, "PROFILE_SCRIPT", script),
            self.assertRaises(bc.BuildFailed) as e,
        ):  # no .profraw is written: the build says so
            bc.profile_run(Path("x.so"), Path("x.npy"), raw, "main")
        self.assertIn("NUMPY-OK", str(e.exception))
        self.assertNotIn("No module named", str(e.exception))


class CompareWorkerIsolationTest(unittest.TestCase):
    """The timing workers of compare.py (and of `--backend auto`) are `python -I` children too: they get the parent's site folders after `--worker SPEC`."""

    def test_the_worker_finds_numpy_only_with_the_parents_site_folders_under_python_I(self):
        """`python -I -S` has no site-packages at all: the worker finds numpy only if the folders are passed."""
        script = str(HERE.parent / "scripts" / "compare.py")
        tmp = Path(tempfile.mkdtemp(prefix="qf-worker-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        run = lambda *extra: subprocess.run(  # noqa: E731
            [sys.executable, "-I", "-S", script, "--worker", str(tmp / "no-such-spec.json"), *extra], capture_output=True, text=True
        )
        self.assertIn("No module named 'numpy'", run().stderr)
        got = run(json.dumps(bc.parent_site_paths()))
        self.assertNotIn("No module named 'numpy'", got.stderr)
        self.assertIn("no-such-spec.json", got.stderr)  # it got as far as reading the spec

    def test_the_parent_builds_the_worker_command_with_the_folders(self):
        source = (HERE.parent / "scripts" / "compare.py").read_text()
        self.assertRegex(source, r'"--worker",\s*str\(work[^\n]*\),\s*json\.dumps\(backend_codegen\.parent_site_paths\(\)\)')


if __name__ == "__main__":
    unittest.main()
