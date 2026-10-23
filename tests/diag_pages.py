#!/usr/bin/env python3
"""Diagnostic, not a test: WHICH pages of a model file stay mapped while `select_stream.ModelFile` reads it.

`tests/test_scale.py` checks that the pages of a mapped model file leave the resident set as they are read (the product lets go of them with
`madvise(MADV_DONTNEED)` behind the cursor). When that check fails on a machine nobody can log into (a CI runner), this script says why:

  * the machine: kernel, file system of the temp folder, transparent huge pages, read-ahead, library versions, the page size;
  * the model: the byte span of every key of the file down to the list of trees, every array above 64 KB, and the keys of a tree;
  * the residency: /proc/self/pagemap (no root needed) gives the "present" bit of every page of the mapping, printed as file-offset ranges
    after `ModelFile(...)` and at four points of `trees()`, with the share of each key of the file that is resident;
  * every `madvise` the product issues (offset, length, outcome) and how many bytes of that range are still resident right after it;
  * a CONTROL: the same file read by a plain loop that touches one byte a page and lets go behind itself on the same schedule. If the control
    also keeps pages, the machine does not give pages back; if only `ModelFile` keeps them, the product reads or maps something it should not.

usage: python tests/diag_pages.py [--trees N] [--model FILE]      (builds the test's model with xgboost unless --model names one)
It never fails (exit 0): CI runs it before the unit tests as a report. Linux only; elsewhere it says so."""

from __future__ import annotations

import mmap
import os
import platform
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import select_stream as ss

PAGE = mmap.PAGESIZE
WINDOW = 1 << 18  # the release window the unit test uses
NF = 8


def out(*a):
    print(*a, flush=True)


def text(path):
    try:
        return Path(path).read_text().strip().replace("\n", " ")
    except Exception as e:
        return f"<{type(e).__name__}>"


# ----------------------------------------------------------------------------------------------------------------------- machine
def mount_of(path):
    """(mount point, file system type, source) of the mount that holds `path`: the longest mount point that is a prefix of it."""
    real, best = os.path.realpath(path), ("?", "?", "?")
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mp = left.split()[4].replace("\\040", " ")
            fs = right.split()
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and (best[0] == "?" or len(mp) >= len(best[0])):
                best = (mp, fs[0], fs[1] if len(fs) > 1 else "?")
    except Exception as e:
        best = ("?", f"<{type(e).__name__}>", "?")
    return best


def machine(folder):
    out("== machine")
    out(
        f"kernel {platform.release()}  arch {platform.machine()}  python {platform.python_version()}  numpy {np.__version__}  page size {PAGE}"
    )
    try:
        import xgboost

        out(f"xgboost {xgboost.__version__}")
    except Exception as e:
        out(f"xgboost not importable ({type(e).__name__})")
    mp, fs, src = mount_of(folder)
    out(f"temp folder {folder}: mount {mp}  type {fs}  source {src}")
    thp = "/sys/kernel/mm/transparent_hugepage"
    out(f"THP enabled [{text(thp + '/enabled')}]  defrag [{text(thp + '/defrag')}]  shmem [{text(thp + '/shmem_enabled')}]")
    st = os.stat(folder)
    dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    ra = text(f"/sys/dev/block/{dev}/queue/read_ahead_kb")
    if ra.startswith("<"):
        ra = text(f"/sys/class/bdi/{dev}/read_ahead_kb")
    out(f"device {dev}  read_ahead_kb {ra}  fault_around_bytes {text('/sys/kernel/debug/fault_around_bytes')}")
    mem = [ln for ln in text("/proc/meminfo").split("kB") if re.search(r"MemTotal|FileHugePages|FilePmdMapped|ShmemPmdMapped", ln)]
    out("meminfo " + " | ".join(ln.strip() for ln in mem))


# ----------------------------------------------------------------------------------------------------------------------- the file
def iter_object(buf, pos, typ=None):
    """The keys of the UBJSON object at `pos` (its marker unless the container fixed the type `typ`) -> (key, value start, value end, kind)."""
    if typ is None:
        pos += 1
    sub, cnt, p = ss._header(buf, pos)
    i = 0
    while cnt is None or i < cnt:
        if cnt is None and buf[p] == ss._OBJ_END:
            return
        n, p = ss._int_at(buf, p)
        key = bytes(buf[p : p + n]).decode("utf-8", "replace")
        p += n
        end = ss._skip(buf, p, sub)
        yield key, p, end, chr(sub if sub is not None else buf[p])
        p, i = end, i + 1


def iter_array(buf, pos):
    """The elements of the UBJSON array at `pos` (its marker) -> (start, end, element type or None)."""
    sub, cnt, p = ss._header(buf, pos + 1)
    i = 0
    while cnt is None or i < cnt:
        if cnt is None and buf[p] == ss._ARR_END:
            return
        end = ss._skip(buf, p, sub)
        yield p, end, sub
        p, i = end, i + 1


def layout(buf):
    """Print the byte span of the keys down to the list of trees (and of every array above 64 KB). -> (span of the list of trees,
    {key: [(start, end)...]} for the keys of every tree, [(label, start, end)] regions)."""
    path_to_trees = ("learner", "gradient_booster", "model", "trees")
    is_ubj = buf[:1] == b"{" and buf[1:2] not in (b'"', b" ", b"\n")
    out(f"== layout of the file ({len(buf):,} bytes, ubj: {is_ubj})")
    found = {}

    def walk(pos, path, depth):
        on_path = path == path_to_trees[: len(path)]
        for key, a, b, kind in iter_object(buf, pos):
            here = (*path, key)
            if here == path_to_trees:
                found["trees"] = (a, b)
            if on_path or b - a > 65536:
                note = ""
                if kind == "[":
                    sub, cnt, _ = ss._header(buf, a + 1)
                    note = f" array of {chr(sub) if sub else 'mixed'} x {cnt}"
                out(f"{'  ' * depth}{'.'.join(here)}: {kind} [{a},{b}) {b - a:,} bytes{note}")
            if kind == "{" and depth < 6:
                walk(a, here, depth + 1)

    walk(0, (), 0)
    per_key = {}
    if "trees" not in found:
        out("no list of trees found at learner.gradient_booster.model.trees")
        return None, per_key, []
    a, b = found["trees"]
    regions = [("before the trees", 0, a), ("the trees", a, b), ("after the trees", b, len(buf))]
    n = 0
    for s, _e, sub in iter_array(buf, a):
        for key, ka, kb, _kind in iter_object(buf, s, sub):
            per_key.setdefault(key, []).append((ka, kb))
        n += 1
    first = [(k, v[0][1] - v[0][0]) for k, v in per_key.items()]
    out(f"trees: {n}; bytes per key of the first tree: " + ", ".join(f"{k} {v}" for k, v in first))
    return (a, b), per_key, regions


# ------------------------------------------------------------------------------------------------------------------------- residency
class Residency:
    """What /proc/self/pagemap says about the mappings of one file (by inode)."""

    def __init__(self, path, per_key, regions):
        self.path, self.ino, self.size = str(path), os.stat(path).st_ino, os.stat(path).st_size
        self.per_key = {k: (np.array([s for s, _ in v]), np.array([e for _, e in v])) for k, v in per_key.items()}
        self.regions = regions
        self.npages = (self.size + PAGE - 1) // PAGE
        self.ok = True

    def vmas(self):
        found = []
        with open("/proc/self/maps") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 5 and f[4] == str(self.ino):
                    a, b = (int(x, 16) for x in f[0].split("-"))
                    found.append((a, b, int(f[2], 16)))
        return found

    def present(self):
        """bool per page of the FILE: mapped and present in some mapping of this process."""
        res = np.zeros(self.npages, dtype=bool)
        with open("/proc/self/pagemap", "rb") as fh:
            for a, b, off in self.vmas():
                fh.seek(a // PAGE * 8)
                raw = fh.read((b - a) // PAGE * 8)
                bits = (np.frombuffer(raw, dtype="<u8") >> 63).astype(bool)
                first = off // PAGE
                k = min(len(bits), self.npages - first)
                res[first : first + k] |= bits[:k]
        return res

    def smaps(self):
        """The Rss and the huge-page counters of the mappings of the file, from smaps."""
        got, inside = {}, False
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                    inside = line.split()[4] == str(self.ino)
                elif inside:
                    k, _, v = line.partition(":")
                    if k in ("Rss", "Private_Clean", "Shared_Clean", "FilePmdMapped", "THPeligible"):
                        got[k] = got.get(k, 0) + int(v.split()[0])
                    elif k == "VmFlags":
                        got["VmFlags"] = v.strip()
        return got

    def in_range(self, res, a, b):
        """Resident bytes of [a, b) of the file by whole pages."""
        pa, pb = a // PAGE, (b + PAGE - 1) // PAGE
        return int(res[pa:pb].sum()) * PAGE

    def runs(self, res=None):
        """The stretches of the file that are resident, as [(start byte, end byte)]."""
        res = self.present() if res is None else res
        runs, i = [], 0
        while i < self.npages:
            if res[i]:
                j = i
                while j < self.npages and res[j]:
                    j += 1
                runs.append((i * PAGE, min(j * PAGE, self.size)))
                i = j
            else:
                i += 1
        return runs

    def report(self, label, extra="", full=True):
        try:
            res = self.present()
        except Exception as e:
            self.ok = False
            out(f"-- {label}: no pagemap ({type(e).__name__}: {e}); smaps {self.smaps()}")
            return None
        total = int(res.sum()) * PAGE
        runs = self.runs(res)
        sm = self.smaps()
        out(f"-- {label}{extra}: {total / 2**20:.2f} MB of {self.size / 2**20:.2f} MB resident in {len(runs)} runs (smaps {sm})")
        if not full:
            return total
        big = sorted(runs, key=lambda r: r[0] - r[1])[:8]
        out("   largest runs [byte start..end): " + ", ".join(f"[{a}..{b})={(b - a) // 1024}K" for a, b in sorted(big)))
        tenth = self.size / 10
        out(
            "   per tenth of the file (KB): "
            + " ".join(str(self.in_range(res, int(k * tenth), int((k + 1) * tenth)) // 1024) for k in range(10))
        )
        out("   regions: " + "; ".join(f"{n} {self.in_range(res, a, b) / 1024:.0f} of {(b - a) / 1024:.0f} KB" for n, a, b in self.regions))
        if self.per_key:
            cs = np.concatenate([[0], np.cumsum(np.repeat(res, PAGE)[: self.size], dtype=np.int64)])
            parts = []
            for k, (s, e) in self.per_key.items():
                parts.append(f"{k} {int((cs[e] - cs[s]).sum()) // 1024}/{int((e - s).sum()) // 1024}K")
            out("   resident / bytes per key of the trees: " + ", ".join(parts))
        return total


def describe_residency(path):
    """One line for a test failure: which stretches of the file are resident now (file offsets in KB) and what smaps says of the huge pages."""
    try:
        res = Residency(path, {}, [])
        runs = res.runs()
        big = sorted(runs, key=lambda r: r[0] - r[1])[:6]
        sm = res.smaps()
        shown = ", ".join(f"[{a >> 10}K..{b >> 10}K)" for a, b in sorted(big))
        return (
            f"{len(runs)} resident stretches, the largest {shown}; smaps Rss {sm.get('Rss')} kB, FilePmdMapped {sm.get('FilePmdMapped')} kB"
        )
    except Exception as e:
        return f"(no residency detail: {type(e).__name__}: {e})"


# ------------------------------------------------------------------------------------------------------------------------- scenarios
def model_file(args):
    if "--model" in args:
        return Path(args[args.index("--model") + 1]), False
    import xgboost as xgb

    trees = int(args[args.index("--trees") + 1]) if "--trees" in args else 1500
    x = np.random.default_rng(2).normal(size=(3000, NF)).astype(np.float32)
    dmat = xgb.DMatrix(x, (x[:, 0] * x[:, 1] > 0).astype(int))
    path = Path(tempfile.mkdtemp(prefix="diag_pages_")) / "pages.ubj"
    params = {"tree_method": "hist", "max_depth": 7, "objective": "binary:logistic", "num_parallel_tree": trees, "subsample": 0.9}
    xgb.train({**params, "learning_rate": 1, "nthread": 2}, dmat, num_boost_round=1).save_model(str(path))
    return path, True


def probe_write_patterns(folder):
    """What one page fault maps for a file in the page cache, by how it got there: the kernel may hold a file written in one go in
    folios of up to 2 MB, one written in pieces in smaller ones, and one read back from disk in what read-ahead made."""
    out("== what one page fault maps (KB at 20/35/50/65/80 % of an 8 MB file), by how the file came to be in the page cache")
    blob = np.random.default_rng(5).bytes(8 << 20)
    cases = (
        ("one write of 8 MB", 8 << 20, False),
        ("8 writes of 1 MB", 1 << 20, False),
        ("128 writes of 64 KB", 64 << 10, False),
        ("one write, then dropped from the cache and read back by the faults", 8 << 20, True),
    )
    for label, piece, dropped in cases:
        probe = Path(tempfile.mkstemp(prefix="diag_unit_", dir=folder)[1])
        try:
            with open(probe, "wb", buffering=0) as fh:
                for k in range(0, len(blob), piece):
                    fh.write(blob[k : k + piece])
                os.fsync(fh.fileno())
                if dropped:
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            res, got = Residency(probe, {}, []), []
            with open(probe, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                    before = int(res.present().sum()) * PAGE
                    mm[int(res.size * frac) + 12345]
                    got.append((int(res.present().sum()) * PAGE - before) >> 10)
                    mm.madvise(mmap.MADV_DONTNEED)
            out(f"   {label}: {got}")
        except Exception as e:
            out(f"   {label}: no result ({type(e).__name__}: {e})")
        finally:
            probe.unlink(missing_ok=True)


def plain_map(fh):
    return mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


def probe_unit(path, res, mapper, label):
    """What one page fault maps: touch one byte at a few offsets of a fresh mapping, let the mapping go in between."""
    got, pmd = [], []
    with open(path, "rb") as fh:
        mm = mapper(fh)
        try:
            for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                before = int(res.present().sum()) * PAGE
                mm[int(res.size * frac) + 12345]
                got.append((int(res.present().sum()) * PAGE - before) >> 10)
                pmd.append(res.smaps().get("FilePmdMapped"))
                mm.madvise(mmap.MADV_DONTNEED)
        finally:
            mm.close()
    out(f"== one page fault maps (KB), touching a byte at 20/35/50/65/80 % of the file, {label}: {got}; smaps FilePmdMapped kB: {pmd}")


def scenario_model_file(path, res, mapper, label):
    out(f"== ModelFile, {label}; release window {WINDOW >> 10} KB (the unit test's), the product's default is {ss.RELEASE_BYTES >> 20} MB")
    calls = []
    original, real_map = ss.ModelFile._madvise, ss.map_file

    def logged(self, start, length):
        err = original(self, start, length)
        try:
            left = res.in_range(res.present(), start, start + length)
        except Exception:
            left = -1
        calls.append((start, length, err, left))
        return err

    ss.RELEASE_BYTES, saved = WINDOW, ss.RELEASE_BYTES
    ss.ModelFile._madvise, ss.map_file = logged, mapper
    try:
        mf = ss.ModelFile(path)
        res.report("after ModelFile(...)", f" (released up to {mf._released}, {len(calls)} madvise calls so far)")
        n = mf.span.count or 0
        marks = {0, n // 3, 2 * n // 3, max(n - 1, 0)}
        for i, (_pos, _t) in enumerate(mf.trees(ss.WANT)):
            if i in marks:
                res.report(f"in trees(), after tree {i} of {n}", f" (released up to {mf._released})", full=i == n // 3)
        res.report("after the last tree", f" (released up to {mf._released})", full=False)
        mf.close()
    finally:
        ss.ModelFile._madvise, ss.map_file = original, real_map
        ss.RELEASE_BYTES = saved
    bad = [c for c in calls if c[2] is not None]
    left = [c for c in calls if c[3] > 0]
    out(f"   madvise calls: {len(calls)}; failed: {len(bad)}; ranges with pages still resident right after the call: {len(left)}")
    for start, length, err, resident in calls[:3] + calls[-2:]:
        out(
            f"   madvise(DONTNEED, offset {start}, length {length}) -> {'ok' if err is None else repr(err)}; still resident in it: {resident} bytes"
        )


def scenario_control(path, res, mapper, label):
    out(f"== control, {label}: one byte a page read in order, pages let go of behind on the same {WINDOW >> 10} KB schedule")
    with open(path, "rb") as fh:
        mm = mapper(fh)
        try:
            released = 0
            for off in range(0, len(mm), PAGE):
                mm[off]
                end = off - off % PAGE
                if end - released >= WINDOW:
                    mm.madvise(mmap.MADV_DONTNEED, released, end - released)
                    released = end
            res.report("control, after the last page", f" (released up to {released})", full=False)
        finally:
            mm.close()


def main(argv):
    if not sys.platform.startswith("linux"):
        out("diag_pages.py: Linux only (/proc/self/pagemap)")
        return
    path, made = model_file(argv)
    try:
        machine(path.parent)
        buf = path.read_bytes()  # a plain read: it maps nothing in this process
        _span, per_key, regions = layout(buf)
        del buf
        res = Residency(path, per_key, regions)
        probe_write_patterns(path.parent)
        for label, mapper in (
            ("mapped as shipped (MADV_NOHUGEPAGE asked)", ss.map_file),
            ("mapped plainly (no MADV_NOHUGEPAGE)", plain_map),
        ):
            probe_unit(path, res, mapper, label)
            scenario_model_file(path, res, mapper, label)
            scenario_control(path, res, mapper, label)
    finally:
        if made:
            try:
                path.unlink()
                path.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        traceback.print_exc()
    sys.exit(0)
