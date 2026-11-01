# Tests

    python -I -m pytest -q -p no:cacheprovider tests/<file>.py

The engine tests (`tests/test_vector_flat.py::EngineTest`) need a library that reads format 7. A fresh worktree ships a
pre-format-7 `bin/libtreeflat-x86_64.so`: build the current one first with `rust/build.sh` (`QUICK=1 rust/build.sh` skips fmt, clippy
and the Rust tests). With a stale library those tests SKIP by default; set `QF_REQUIRE_FORMAT7=1` and they FAIL loudly instead
(use it in every release-gate and review run).
