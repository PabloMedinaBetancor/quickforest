#!/usr/bin/env python3
"""Greedy forest selection on xgboost models too big to hold in memory (hundreds of thousands of trees).

`prune.py` normally loads the model with xgboost and holds the leaf values of the selection rows, a (trees x rows) float32 matrix, in
memory: 400,000 trees x 10,000 rows is 16 GB, and the model parsed needs a good deal more. This module makes the same selection with memory
that does not grow with the number of trees:

  1. SCAN. The model file (.json or .ubj) is read once, tree by tree, straight from the file (never loaded as a whole): each tree is
     checked as `qf_common.parse_model` checks it, and its structure is written to flat files in a scratch folder.
  2. WALK. Trees are taken in chunks and walked over the selection rows by a numpy walker (xgboost's own rule: `x < split` goes left, a
     missing value goes the way `default_left` says). What is kept is one small code per (tree, row), the leaf the row ends in (1 byte for
     trees of up to 256 leaves), plus one table of leaf values per tree: a quarter or less of the float32 matrix. When the codes do not fit
     half of the memory budget they live in a file, read back a block of trees at a time.
  3. SELECT. `prune.greedy_order` runs on a `LeafCodes` object that hands out the float32 rows of the candidate trees it asks for, so the
     selection is the one it makes on the full matrix (same numbers, same order): `tests/test_scale.py` pins it.
  4. WRITE. The kept trees are read again from the model file by their position and written as a normal model.

Used by `prune.run` when the in-memory path would not fit `--memory-mb` (`wanted`). A `StreamSource` answers the calls `prune.run` makes on
an `XgbSource`. xgboost models only: LightGBM and YDF files are read by their libraries as before."""

from __future__ import annotations

import copy
import itertools
import json
import mmap
import os
import re
import shutil
import struct
import tempfile
import time
import weakref
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qf_common as qf
import qf_formats

DEFAULT_MEMORY_MB = 1500
HELD_ROWS = 20000  # held-out rows the full forest answers in a streamed run unless told otherwise (each costs one walk of every tree)
CHUNK_CELLS = 1 << 19  # (trees x rows) cells the walker handles at once: its temporaries take about 50 bytes a cell
RELEASE_BYTES = 32 << 20  # the model file's pages are let go of in steps of this many bytes as they are read
SMALL_ARRAY = (
    100_000  # a UBJSON array of the model's header longer than this stays a numpy array (`tree_info` of a forest of a million trees)
)
WANT = ("left_children", "right_children", "split_indices", "split_conditions", "default_left", "split_type", "categories", "tree_param")


class ScaleError(ValueError):
    """A model file that cannot be read in a stream (truncated, or not what its format says)."""


# ----------------------------------------------------------------------------------------------------------------------- UBJSON
_INT = {ord("i"): ("b", 1), ord("U"): ("B", 1), ord("I"): (">h", 2), ord("l"): (">i", 4), ord("L"): (">q", 8)}
_NUM = {ord("i"): ">i1", ord("U"): ">u1", ord("I"): ">i2", ord("l"): ">i4", ord("L"): ">i8", ord("d"): ">f4", ord("D"): ">f8"}
_WIDTH = {ord("i"): 1, ord("U"): 1, ord("I"): 2, ord("l"): 4, ord("L"): 8, ord("d"): 4, ord("D"): 8}
_T, _F, _Z, _N, _S, _C = (ord(c) for c in "TFZNSC")
_OBJ, _OBJ_END, _ARR, _ARR_END, _TYPE, _COUNT = (ord(c) for c in "{}[]$#")


def _int_at(buf, pos):
    """A UBJSON integer (marker and value) -> (value, next position)."""
    f = _INT.get(buf[pos])
    if f is None:
        raise ScaleError(f"UBJSON: an integer was expected at byte {pos}")
    return struct.unpack_from(f[0], buf, pos + 1)[0], pos + 1 + f[1]


def _header(buf, pos):
    """After '[' or '{': the optional `$type` and `#count` -> (type or None, count or None, next position)."""
    typ = cnt = None
    if buf[pos] == _TYPE:
        typ, pos = buf[pos + 1], pos + 2
    if buf[pos] == _COUNT:
        cnt, pos = _int_at(buf, pos + 1)
    elif typ is not None:
        raise ScaleError(f"UBJSON: a typed container without a count at byte {pos}")
    return typ, cnt, pos


def _skip(buf, pos, typ=None):
    """The position after the value at `pos` (its marker is missing when the container fixed the type `typ`)."""
    if typ is None:
        typ, pos = buf[pos], pos + 1
    w = _WIDTH.get(typ)
    if w is not None:
        return pos + w
    if typ in (_T, _F, _Z, _N):
        return pos
    if typ == _C:
        return pos + 1
    if typ == _S:
        n, pos = _int_at(buf, pos)
        return pos + n
    if typ in (_ARR, _OBJ):
        sub, cnt, pos = _header(buf, pos)
        obj = typ == _OBJ
        if cnt is not None and sub in _WIDTH and not obj:
            return pos + cnt * _WIDTH[sub]
        i = 0
        while cnt is None or i < cnt:
            if cnt is None and buf[pos] == (_OBJ_END if obj else _ARR_END):
                return pos + 1
            if obj:
                n, pos = _int_at(buf, pos)
                pos += n
            pos = _skip(buf, pos, sub)
            i += 1
        return pos
    raise ScaleError(f"UBJSON: unsupported marker {chr(typ)!r} at byte {pos - 1}")


def _decode(buf, pos, typ=None, tolist=False, lazy=None, path=(), skipper=_skip):
    """One value -> (python value, next position). Numeric arrays are numpy arrays in native byte order (lists with `tolist`), objects are
    dicts. `lazy`: the key path of an array that is not decoded but returned as a `Span` (the model's trees); `skipper(buf, pos)` finds
    where that array ends."""
    if typ is None:
        typ, pos = buf[pos], pos + 1
    if typ in _NUM:
        dt = np.dtype(_NUM[typ])
        v = np.frombuffer(buf, dtype=dt, count=1, offset=pos)[0].item()
        return v, pos + dt.itemsize
    if typ == _T or typ == _F:
        return typ == _T, pos
    if typ == _Z or typ == _N:
        return None, pos
    if typ == _C:
        return chr(buf[pos]), pos + 1
    if typ == _S:
        n, pos = _int_at(buf, pos)
        return bytes(buf[pos : pos + n]).decode("utf-8"), pos + n
    if typ == _ARR:
        sub, cnt, pos = _header(buf, pos)
        if cnt is not None and sub in _NUM:
            arr = np.frombuffer(buf, dtype=np.dtype(_NUM[sub]), count=cnt, offset=pos).astype(np.dtype(_NUM[sub]).newbyteorder("="))
            return (arr.tolist() if tolist is True or (tolist and cnt <= SMALL_ARRAY) else arr), pos + cnt * _WIDTH[sub]
        out, i = [], 0
        while cnt is None or i < cnt:
            if cnt is None and buf[pos] == _ARR_END:
                pos += 1
                break
            v, pos = _decode(buf, pos, sub, tolist)
            out.append(v)
            i += 1
        return out, pos
    if typ == _OBJ:
        sub, cnt, pos = _header(buf, pos)
        out, i = {}, 0
        while cnt is None or i < cnt:
            if cnt is None and buf[pos] == _OBJ_END:
                pos += 1
                break
            n, pos = _int_at(buf, pos)
            key = bytes(buf[pos : pos + n]).decode("utf-8")
            pos += n
            sub_path = (*path, key)
            if lazy is not None and sub_path == lazy and sub is None and buf[pos] == _ARR:
                atyp, acnt, first = _header(buf, pos + 1)
                end = skipper(buf, pos)
                out[key], pos, i = Span(first, acnt, atyp, end), end, i + 1
                continue
            out[key], pos = _decode(buf, pos, sub, tolist, lazy, sub_path, skipper)
            i += 1
        return out, pos
    raise ScaleError(f"UBJSON: unsupported marker {chr(typ)!r} at byte {pos - 1}")


@dataclass
class Span:
    """Where the trees of a UBJSON model are: the position of the first one, how many the header counts (None: counted by the end
    marker), and the position after the array."""

    first: int
    count: int | None
    typ: int | None
    end: int = 0


def _tree_object(buf, pos, want=None, tolist=False):
    """One tree object (UBJSON) -> (dict of the wanted keys, next position)."""
    if buf[pos] != _OBJ:
        raise ScaleError(f"UBJSON: a tree object was expected at byte {pos}")
    sub, cnt, pos = _header(buf, pos + 1)
    out, i = {}, 0
    while cnt is None or i < cnt:
        if cnt is None and buf[pos] == _OBJ_END:
            return out, pos + 1
        n, pos = _int_at(buf, pos)
        key = bytes(buf[pos : pos + n]).decode("utf-8")
        pos += n
        if want is None or key in want:
            out[key], pos = _decode(buf, pos, sub, tolist)
        else:
            pos = _skip(buf, pos, sub)
        i += 1
    return out, pos


# ------------------------------------------------------------------------------------------------------------------------- the file
def map_file(fh) -> mmap.mmap:
    """The whole file, read only, mapped the way a stream that lets go of its pages behind it needs. A kernel that holds a file in 2 MB
    folios of its page cache (ext4 and xfs on recent Linux) can map ALL 2 MB of one on a fault in it, and map it again on the next fault
    after `madvise(MADV_DONTNEED)` split that mapping: the pages just let go of come back at once, and the resident set cannot go under
    2 MB plus the release window. MADV_NOHUGEPAGE asks for small mappings (64 KB a fault) instead; where the kernel ignores it the bound is
    the same, and it still does not grow with the file (tests/diag_pages.py prints what a machine does, tests/test_scale.py measures it)."""
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        mm.madvise(mmap.MADV_NOHUGEPAGE)
    except (AttributeError, OSError, ValueError):  # not Linux, or no huge-page support: nothing to ask for
        pass
    return mm


_TREES_KEY = re.compile(rb'"trees"\s*:\s*\[')
_JSON = json.JSONDecoder()
_SPACE = " \t\r\n,"


class ModelFile:
    """An xgboost model file (.json or .ubj) read in a stream: `meta` is the document without its trees, `trees(want)` yields
    (position, tree dict) one at a time, `tree_at(position)` reads one tree again."""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = None
        try:
            self._fh = open(self.path, "rb")
            self.mm = map_file(self._fh)
        except (OSError, ValueError) as e:  # an empty file cannot be mapped
            if self._fh is not None:
                self._fh.close()
            raise ScaleError(f"{self.path.name}: cannot read the model ({getattr(e, 'strerror', None) or e})") from None
        self.size = len(self.mm)
        head = bytes(self.mm[:64]).lstrip()
        self.ubj = head[:1] == b"{" and head[1:2] not in (b'"', b" ", b"\n", b"\r", b"\t", b"}", b"")
        self._json_end = None
        self._released = 0
        try:
            if self.ubj:
                doc, _ = _decode(
                    self.mm, 0, tolist="small", lazy=("learner", "gradient_booster", "model", "trees"), skipper=self._skip_trees
                )
                self.meta = doc
                self.span = doc["learner"]["gradient_booster"]["model"]["trees"]
                if not isinstance(self.span, Span):
                    raise KeyError("trees")
            else:
                self.meta, self.span = None, self._find_json_trees()
        except ScaleError:
            self.close()
            raise
        except (KeyError, IndexError, struct.error, UnicodeDecodeError, ValueError, TypeError, AttributeError) as e:
            self.close()
            raise ScaleError(f"{self.path.name}: not a complete xgboost model ({type(e).__name__}: {str(e)[:100]})") from None

    def close(self):
        for handle in (self.mm, self._fh):
            try:
                handle.close()
            except (BufferError, ValueError):  # a view of the map is still alive: the garbage collector unmaps it
                pass

    def _skip_trees(self, buf, pos: int) -> int:
        """Where the list of trees that starts at `pos` ends, letting go of the pages already passed (the list is most of the file)."""
        sub, cnt, p = _header(buf, pos + 1)
        if cnt is None or sub is not None:
            return _skip(buf, pos)
        for _ in range(cnt):
            p = _skip(buf, p)
            self._release(p)
        return p

    def _release(self, upto: int) -> None:
        """The map's pages before `upto` are not needed again soon: they stay in the system's cache but leave this process's resident set."""
        end = upto - upto % mmap.PAGESIZE
        if end - self._released >= RELEASE_BYTES:
            self._madvise(self._released, end - self._released)
            self._released = end

    def _madvise(self, start: int, length: int):
        """The one call that lets go of pages (tests/diag_pages.py logs it). None when it was made, else the error that stopped it."""
        try:
            self.mm.madvise(mmap.MADV_DONTNEED, start, length)
        except (AttributeError, OSError, ValueError) as e:  # no madvise here: the pages go when the system needs them
            return e
        return None

    # ---- JSON
    def _find_json_trees(self) -> int:
        for m in _TREES_KEY.finditer(self.mm):
            if m.start() and self.mm[m.start() - 1] == ord("\\"):
                continue
            try:
                obj, _ = self._json_at(m.end())
            except ScaleError:
                continue
            if isinstance(obj, dict) and "left_children" in obj:
                return m.end()
        raise ScaleError(f"{self.path.name}: no list of trees found (is this an xgboost model?)")

    def _json_at(self, pos: int, win: int = 1 << 20):
        """The JSON value starting at or after `pos` -> (value, absolute end). The window grows until the value fits."""
        while True:
            text = self.mm[pos : pos + win].decode("latin-1")  # the trees are ASCII; latin-1 keeps byte and character positions equal
            i = 0
            while i < len(text) and text[i] in _SPACE:
                i += 1
            try:
                obj, j = _JSON.raw_decode(text, i)
                return obj, pos + j
            except ValueError:
                if pos + win >= self.size:
                    raise ScaleError(f"{self.path.name}: the JSON is cut short or malformed near byte {pos}") from None
                win *= 4

    def _json_trees(self):
        pos, win = self.span, 1 << 24
        while True:
            self._release(pos)
            text = self.mm[pos : pos + win].decode("latin-1")
            n, i, got = len(text), 0, False
            at_end = pos + win >= self.size
            while True:
                while i < n and text[i] in _SPACE:
                    i += 1
                if i >= n:
                    break
                if text[i] == "]":
                    self._json_end = pos + i
                    return
                try:
                    obj, j = _JSON.raw_decode(text, i)
                except ValueError:
                    if at_end:
                        raise ScaleError(f"{self.path.name}: a tree is cut short or malformed near byte {pos + i}") from None
                    break  # the window ends inside this tree: read on from its start
                yield pos + i, obj
                i, got = j, True
            if at_end:
                raise ScaleError(f"{self.path.name}: the list of trees is not closed (truncated?)")
            if i == 0 and not got:
                win *= 4  # one tree is larger than the window
            else:
                pos += i

    def _json_meta(self):
        head = bytes(self.mm[: self.span])
        tail = bytes(self.mm[self._json_end :])
        return json.loads(head + tail)

    # ---- both
    def trees(self, want=WANT):
        """(position, tree) for every tree, in order. JSON: dicts of lists; UBJSON: dicts of numpy arrays. `meta` is complete after the
        last one (for JSON it is read from what stands before and after the trees)."""
        self._released = 0  # the pages are let go of again as this pass goes (the first pass, the skip over the list, let go of them all)
        if self.ubj:
            pos, i = self.span.first, 0
            while self.span.count is None or i < self.span.count:
                if self.span.count is None and self.mm[pos] == _ARR_END:
                    break
                start = pos
                t, pos = _tree_object(self.mm, pos, want)
                yield start, t
                i += 1
                self._release(pos)
            if self.span.count is not None and pos != self.span.end:
                raise ScaleError(f"{self.path.name}: the trees end at byte {pos}, the list at {self.span.end}")
        else:
            try:
                yield from self._json_trees()
            except RecursionError:
                raise ScaleError(f"{self.path.name}: a tree is nested too deeply") from None
            self.meta = self._json_meta()

    def tree_at(self, pos: int) -> dict:
        """One whole tree as plain Python lists, from its position."""
        if self.ubj:
            return _tree_object(self.mm, pos, None, tolist=True)[0]
        return self._json_at(pos)[0]


# ------------------------------------------------------------------------------------------------------------------------ the scan
FIELDS = (("left", "<i4"), ("right", "<i4"), ("feat", "<i4"), ("cond", "<f4"), ("dleft", "|u1"))


class Packed:
    """Every tree of the model in flat arrays on disk: node arrays concatenated in tree order; `node_off[t]` is the first node of tree t,
    `leaf_off[t]` the first leaf (in node order), `max_leaves` the largest count of leaves in a tree. The arrays are read back by
    position, never mapped: their pages do not count as the process's resident memory."""

    def __init__(self, folder: Path, node_off: np.ndarray, leaf_off: np.ndarray):
        self.folder = folder
        self.node_off, self.leaf_off = node_off, leaf_off
        self.n_trees = len(node_off) - 1
        self.max_leaves = int(np.diff(leaf_off).max()) if self.n_trees else 0

    def read(self, name: str, lo: int, hi: int) -> np.ndarray:
        dt = np.dtype(dict(FIELDS)[name])
        return np.fromfile(self.folder / f"{name}.bin", dtype=dt, count=hi - lo, offset=lo * dt.itemsize)

    def close(self):
        pass


def scan(mf: ModelFile, folder: Path):
    """Pass one over the trees: check each, append its arrays to the files of `folder`. Returns (Packed, nodes, max_feature, positions): `nodes` are
    the reachable nodes of every tree, `positions` where each tree starts in the model file. Raises ValueError with the message `qf_common.parse_model` would give."""
    folder.mkdir(parents=True, exist_ok=True)
    outs = {name: open(folder / f"{name}.bin", "wb") for name, _ in FIELDS}
    sizes, reach, leaves, maxfeat, where = (array("q") for _ in range(5))  # 8 bytes a tree each, not a Python int's 36
    pending = {name: [] for name, _ in FIELDS}
    held = 0
    try:
        for i, (at, t) in enumerate(mf.trees(WANT)):
            if np.asarray(t["split_type"]).any() or len(t["categories"]):
                raise ValueError("categorical splits are not supported")
            if int(t["tree_param"].get("size_leaf_vector", "1")) > 1:
                raise ValueError("vector leaves are not supported")
            qf.check_xgb_tree(t, 2**31 - 1, i)  # the feature range is checked against the model's num_feature once the header is known
            left = np.asarray(t["left_children"], np.int32)
            right = np.asarray(t["right_children"], np.int32)
            feat = np.asarray(t["split_indices"], np.int32)
            cond = np.asarray(t["split_conditions"], np.float32)
            dleft = (np.asarray(t["default_left"]) != 0).astype(np.uint8)
            inner = left != -1
            where.append(at)
            sizes.append(len(left))
            leaves.append(int((~inner).sum()))
            reach.append(int(qf.reachable_nodes(left, right).sum()))
            maxfeat.append(int(feat[inner].max()) if inner.any() else -1)
            for name, a in (("left", left), ("right", right), ("feat", feat), ("cond", cond), ("dleft", dleft)):
                pending[name].append(a)
            held += len(left)
            if held >= 1 << 18:
                for name in pending:
                    outs[name].write(np.concatenate(pending[name]).tobytes())
                    pending[name] = []
                held = 0
        for name in pending:
            if pending[name]:
                outs[name].write(np.concatenate(pending[name]).tobytes())
    finally:
        for f in outs.values():
            f.close()
    if not sizes:
        raise ValueError("the model has no trees (trained for zero rounds?)")
    node_off = np.concatenate([[0], np.cumsum(np.frombuffer(sizes, np.int64))]).astype(np.int64)
    leaf_off = np.concatenate([[0], np.cumsum(np.frombuffer(leaves, np.int64))]).astype(np.int64)
    return (
        Packed(folder, node_off, leaf_off),
        np.frombuffer(reach, np.int64),
        np.frombuffer(maxfeat, np.int64),
        np.frombuffer(where, np.int64),
    )


# ------------------------------------------------------------------------------------------------------------------------ the walk
def ordered_map(fn, items, threads: int):
    """`map(fn, items)` on `threads` threads, results in order, never more than twice `threads` of them waiting to be taken."""
    threads = min(threads, 32)
    if threads <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(threads) as pool:
        it = iter(items)
        window = deque(pool.submit(fn, item) for item in itertools.islice(it, 2 * threads))
        while window:
            yield window.popleft().result()
            for item in itertools.islice(it, 1):
                window.append(pool.submit(fn, item))


def auto_threads() -> int:
    """Threads of a streamed run by default. The walk gains from a second thread (1.6x on a 2-core, 4-thread i3) and little from more; the
    selection gains almost nothing from a second one (numpy's calls on blocks this small hold the interpreter lock for a good part of
    their time) and loses a little from four: 48 s with one thread, 50 s with two, 55 s with four (30,000 trees, 10 picks)."""
    return max(1, min(2, os.cpu_count() or 1))


class Rows:
    """The rows of a walk laid out for `walk_trees`: the (rows, features + 1) matrix twice, in the first copy a missing value reads -inf and
    in the second +inf (a split that sends missing values left reads the first copy, one that sends them right the second, so `x < split`
    alone gives xgboost's answer), and a last column of -inf that the leaves read."""

    def __init__(self, x):
        n, f = x.shape
        self.n, self.nf, self.width = n, f, f + 1
        table = np.empty((2, n, f + 1), np.float32)
        table[:, :, f] = -np.inf
        nan = np.isnan(x)
        table[0, :, :f] = np.where(nan, -np.inf, x)
        table[1, :, :f] = np.where(nan, np.inf, x)
        self.table = table.reshape(-1)
        self.half = n * (f + 1)
        self.base = (np.arange(n, dtype=np.intp) * (f + 1))[None, :]


def walk_trees(packed: Packed, a: int, b: int, rows: Rows):
    """Walk trees a..b-1 over `rows`. Returns (codes (b-a, rows) int32: the ordinal of the leaf each row ends in, within its tree; the leaf
    values float32 in the order of the ordinals, concatenated over the trees). A leaf points at itself, so every row takes as many steps as
    the deepest tree needs and rests on its leaf."""
    na, nb = int(packed.node_off[a]), int(packed.node_off[b])
    left, right = packed.read("left", na, nb), packed.read("right", na, nb)
    feat, cond, dleft = packed.read("feat", na, nb), packed.read("cond", na, nb), packed.read("dleft", na, nb).astype(bool)
    sizes = np.diff(packed.node_off[a : b + 1])
    starts = (packed.node_off[a:b] - na).astype(np.intp)
    isleaf = left < 0
    inner = ~isleaf
    me = np.arange(nb - na, dtype=np.intp)
    lg = np.where(isleaf, me, left + np.repeat(starts, sizes)).astype(np.intp)
    # xgboost grows a tree by adding both children of a node together, so the right child is the next node after the left one: the walk
    # then needs one table read for the step, not two (a tree written otherwise takes the general step)
    consecutive = bool((right[inner] == left[inner] + 1).all()) if inner.any() else True
    rg = None if consecutive else np.where(isleaf, me, right + np.repeat(starts, sizes)).astype(np.intp)
    slot = (np.where(isleaf, rows.nf, feat) + np.where(isleaf | dleft, 0, rows.half)).astype(np.intp)
    cw = np.where(isleaf, np.float32(np.inf), cond)
    depth = np.zeros(nb - na, np.int32)
    ii = np.flatnonzero(inner)
    li = lg[ii]
    ri = li + 1 if consecutive else rg[ii]
    while True:  # children come after their parents: the depth settles in as many passes as the tree is deep
        nd = depth[ii] + 1
        if (depth[li] == nd).all() and (depth[ri] == nd).all():
            break
        depth[li], depth[ri] = nd, nd
    pos = np.broadcast_to(starts[:, None], (b - a, rows.n)).copy()
    for _ in range(int(depth.max())):
        v = rows.table.take(rows.base + slot.take(pos, mode="clip"), mode="clip")
        nxt = lg.take(pos, mode="clip")
        if consecutive:
            nxt += ~(v < cw.take(pos, mode="clip"))
        else:
            nxt = np.where(v < cw.take(pos, mode="clip"), nxt, rg.take(pos, mode="clip"))
        pos = nxt
    cs = np.cumsum(isleaf, dtype=np.int32)
    before = cs[starts] - isleaf[starts]
    ordinal = cs - 1 - np.repeat(before, sizes)
    return ordinal.take(pos, mode="clip"), cond[isleaf]


class FileRows:
    """A (rows, columns) matrix of small integers kept in a file and read back with positional reads (`pread`): like the packed arrays, its
    pages never count as the process's resident memory, and several threads can read at once."""

    def __init__(self, path: Path, shape, dtype):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.row_bytes = self.shape[1] * self.dtype.itemsize
        self.nbytes = self.shape[0] * self.row_bytes
        self.fd = None
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)

    def write(self, first: int, block: np.ndarray) -> None:
        data = memoryview(np.ascontiguousarray(block, dtype=self.dtype)).cast("B")
        done, at = 0, first * self.row_bytes
        while done < len(data):
            done += os.pwrite(self.fd, data[done:], at + done)

    def _read(self, buf: np.ndarray, offset: int) -> None:
        view = memoryview(buf).cast("B")
        done = 0
        while done < len(view):
            data = os.pread(self.fd, len(view) - done, offset + done)
            if not data:
                raise OSError("the file of leaf codes is shorter than it was written")
            view[done : done + len(data)] = data
            done += len(data)

    def __getitem__(self, ids):
        if isinstance(ids, (int, np.integer)):
            return self[np.array([ids])][0]
        ids = np.asarray(ids)
        out = np.empty((len(ids), self.shape[1]), self.dtype)
        if len(ids):
            cut = np.flatnonzero(np.diff(ids) != 1) + 1  # one read for each run of consecutive rows
            for lo, hi in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(ids)]]), strict=True):
                self._read(out[lo:hi], int(ids[lo]) * self.row_bytes)
        return out

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.close()


class LeafCodes:
    """The (trees, rows) float32 leaf-value matrix `prune.greedy_order` asks for, held as one small code per (tree, row) plus the leaf values
    of each tree. `lm[i]` is the row of tree i and `lm[ids]` the rows of those trees, both exactly the float32 values the full matrix would
    hold. `margin` is the sum of the leaves per output group over the selection rows (float64, trees added in order, as the full-matrix sum),
    `extra_margin` the same over the extra rows given to the walk."""

    def __init__(self, codes, lv, leaf_off, margin, extra_margin, folder=None):
        self.codes, self.lv = codes, lv
        self.leaf_off = leaf_off.astype(np.int32 if len(lv) < 2**31 - 1 else np.int64)  # int32 indexes: half the memory of a block read
        self.shape = codes.shape
        self.margin, self.extra_margin = margin, extra_margin
        self.on_disk = isinstance(codes, FileRows)
        self._folder = folder

    @property
    def code_bytes(self) -> int:
        return int(self.codes.nbytes)

    def __getitem__(self, ids):
        if isinstance(ids, (int, np.integer)):
            return self.lv[int(self.leaf_off[ids]) + self.codes[ids].astype(np.int64)]
        ids = np.asarray(ids)
        return self.lv.take(self.leaf_off[ids][:, None] + self.codes[ids], mode="clip")  # the indexes are valid by construction

    def close(self):
        if isinstance(self.codes, FileRows):
            self.codes.close()
        self.codes = None
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None


def code_dtype(max_leaves: int):
    return np.uint8 if max_leaves <= 256 else np.uint16 if max_leaves <= 65536 else np.uint32


class StreamSource:
    """The calls `prune.run` makes on an `XgbSource`, for a model that is read in a stream (see the module doc)."""

    fmt = "xgboost"
    unit_of = None
    linear = False

    def __init__(self, path, scratch=None, memory_mb: int = DEFAULT_MEMORY_MB, chunk_cells: int | None = None, threads: int = 1):
        self.path = Path(path)
        self.name = self.path.name
        self.memory_mb = max(1, int(memory_mb))
        self.threads = max(1, int(threads))
        # the walk's temporaries are about 50 bytes a (tree, row) cell, one set per thread: a quarter of the budget, never above CHUNK_CELLS
        self.chunk_cells = max(1, int(chunk_cells or min(CHUNK_CELLS, (self.memory_mb << 20) // (4 * 50 * self.threads))))
        self.scratch = Path(tempfile.mkdtemp(prefix="qf-scratch-")) if scratch is None else Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        weakref.finalize(self, shutil.rmtree, str(self.scratch), True)  # the working files go even when a run ends in an error
        self.timings = {}
        self.packed = None
        t0 = time.perf_counter()
        self.mf = ModelFile(self.path)
        try:
            if self.mf.ubj:
                qf.model_header(
                    self.mf.meta
                )  # the objective, the booster, dart weights: refused before the trees are read (JSON: after them)
            self.packed, nodes, maxfeat, self.tree_pos = scan(self.mf, self.scratch / "packed")
            self.timings["scan"] = round(time.perf_counter() - t0, 2)
            meta = self.mf.meta
            self.spec = qf.spec_from_header(meta, nodes, self.packed.node_off[:-1], self.packed.n_trees)
            bad = np.flatnonzero(maxfeat >= self.spec.n_features)
            if len(bad):
                raise ValueError(f"tree {int(bad[0])}: a split reads a feature the model does not have")
            self.notes = qf_formats.early_stopping_notes(meta, self.name)
            self.size_bytes = self.mf.size
        except BaseException:
            self.close()
            raise

    # -- the walk
    def _rows_per_chunk(self, n_rows: int) -> int:
        return max(1, self.chunk_cells // max(1, n_rows))

    def leaf_matrix(self, x, extra=None, code_budget_mb: float | None = None) -> LeafCodes:
        """Walk the forest over the rows `x` (and `extra`, whose leaves are only summed) -> LeafCodes. The codes stay in memory when they
        fit half the memory budget (or `code_budget_mb`), else in a file of the scratch folder."""
        qf.check_width(x, self.spec.n_features, self.name)
        spec, packed = self.spec, self.packed
        t_n, n = packed.n_trees, len(x)
        dt = code_dtype(packed.max_leaves)
        budget = (self.memory_mb / 2 if code_budget_mb is None else code_budget_mb) * 2**20
        folder = None
        if t_n * n * np.dtype(dt).itemsize <= budget:
            codes = np.empty((t_n, n), dt)
        else:
            folder = self.scratch / "codes"
            folder.mkdir(parents=True, exist_ok=True)
            codes = FileRows(folder / "codes.bin", (t_n, n), dt)
        both = np.ascontiguousarray(x, dtype=np.float32) if extra is None else np.vstack([x, extra]).astype(np.float32, copy=False)
        n_all, rows = len(both), Rows(both)
        lv = np.empty(int(packed.leaf_off[-1]), np.float32)
        margin = np.zeros((spec.groups, n), np.float64)
        extra_margin = None if extra is None else np.zeros((spec.groups, n_all - n), np.float64)
        per = self._rows_per_chunk(n_all)

        def spans():
            a = 0
            while a < t_n:
                b = min(t_n, a + per)
                while (
                    b - a > 1 and int(packed.node_off[b] - packed.node_off[a]) > 4 * self.chunk_cells
                ):  # trees of very many nodes: fewer at once
                    b = a + max(1, (b - a) // 2)
                yield a, b
                a = b

        def work(span):
            return span, walk_trees(packed, span[0], span[1], rows)

        t0 = time.perf_counter()
        for (a, b), (ordinal, vals) in ordered_map(work, spans(), self.threads):
            lo, hi = int(packed.leaf_off[a]), int(packed.leaf_off[b])
            lv[lo:hi] = vals
            if isinstance(codes, FileRows):
                codes.write(a, ordinal[:, :n])
            else:
                codes[a:b] = ordinal[:, :n]
            leaf = lv.take(packed.leaf_off[a:b][:, None] + ordinal, mode="clip")  # (b-a, n_all) float32
            for i in range(b - a):  # trees added one by one in their order: the sum the full matrix gives, bit for bit
                g = spec.tree_info[a + i]
                margin[g] += leaf[i, :n]
                if extra_margin is not None:
                    extra_margin[g] += leaf[i, n:]
        self.timings["leaf_matrix"] = round(time.perf_counter() - t0, 2)
        return LeafCodes(codes, lv, packed.leaf_off[:-1], margin.T, None if extra_margin is None else extra_margin.T, folder)

    def predict(self, x, margin=False, base_margin=None):
        """The full model's answer on the rows `x`, from the walker: float64 sums of the leaves in tree order, rounded to float32 (xgboost
        sums in float32). Used for the held-out rows of a model too big to load with xgboost."""
        lm = self.leaf_matrix(np.zeros((0, self.spec.n_features), np.float32), extra=np.asarray(x, np.float32))
        m = self.spec.base[None, :] + lm.extra_margin
        if base_margin is not None:
            m = m + np.asarray(base_margin, np.float64).reshape(len(x), -1)
        lm.close()
        m = m.astype(np.float32)
        out = m if margin else qf.transform(m, self.spec.kind)
        return out.reshape(len(x), -1).astype(np.float32)

    def set_threads(self, n: int) -> None:
        pass

    def save_pruned(self, kept, mean_scale: bool, dest: Path) -> None:
        keep = sorted({int(i) for g in kept for i in g})
        by_pos = {i: self.mf.tree_at(int(self.tree_pos[i])) for i in keep}
        doc = copy.deepcopy(self.mf.meta)  # `prune_doc` rewrites the document: every rung of `--keep auto` starts from the model's own
        model = doc["learner"]["gradient_booster"]["model"]
        model["trees"] = []
        pb = qf.xgb().Booster()
        pb.load_model(bytearray(qf_formats.prune_doc(doc, self.spec, kept, mean_scale, by_pos)))
        qf.replace_via_temp(dest, lambda tmp: pb.save_model(qf.library_path(tmp)))

    def close(self):
        if self.packed is not None:
            self.packed.close()
        self.mf.close()
        shutil.rmtree(self.scratch, ignore_errors=True)


def _num_trees(path: Path):
    """The model's tree count from its header, without loading the file (None when it cannot be found)."""
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        i = mm.find(b"num_trees")
        if i < 0:
            return None
        tail = bytes(mm[i + 9 : i + 80])
        m = re.match(rb'"\s*:\s*"(\d+)"', tail)  # JSON
        if m:
            return int(m.group(1))
        if tail[:1] == b"S":  # UBJSON string: marker, length, digits
            n, p = _int_at(tail, 1)
            return int(tail[p : p + n]) if tail[p : p + n].isdigit() else None
    return None


def wanted(path, n_rows: int, memory_mb: int) -> bool:
    """Whether the in-memory path would need more than `memory_mb` for this xgboost file and this many selection rows: the float32 leaf
    matrix (4 bytes a tree and row), about 250 MB of working space, and the model parsed (about 10 times its JSON size; a .ubj is read as
    JSON too). Measured on 10,000 trees x 10,000 rows of a binary forest: 1.0 GB peak."""
    p = Path(path)
    if p.suffix not in (".json", ".ubj") or not p.is_file():
        return False
    size = p.stat().st_size
    try:
        trees = _num_trees(p)
    except (OSError, ValueError):
        trees = None
    trees = trees if trees is not None else size / 2000
    as_json = size * (1.33 if p.suffix == ".ubj" else 1.0)
    return 4 * trees * n_rows + (250 << 20) + 10 * as_json > memory_mb * 2**20
