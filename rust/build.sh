#!/bin/bash
# Builds the evaluator for baseline x86-64 (no target-cpu=native: the same .so must run on CPUs without AVX) and copies it to ../bin.
# Any architecture (x86_64, aarch64): the library is named libtreeflat-$(uname -m).so. QUICK=1 skips fmt, clippy and the tests (a user building on an ARM box).
# Usage: rust/build.sh [OUT_DIR]. Uses `cargo` from PATH, or the toolchain under $RUST_DIR/{rustup,cargo} (first toolchain found).
# The build directory is $CARGO_TARGET_DIR, else $TMPDIR/target-treeflat, else rust/target. Nothing is written in the home folder.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ -n "${RUST_DIR:-}" ]; then
  tc=$(ls -d "$RUST_DIR"/rustup/toolchains/*/ | head -1)
  export PATH=${tc}bin:$PATH CARGO_HOME=$RUST_DIR/cargo RUSTUP_HOME=$RUST_DIR/rustup
fi
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-${TMPDIR:+$TMPDIR/target-treeflat}}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$here/target}
out=${1:-$here/../bin}
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
cd "$here"
if [ -z "${QUICK:-}" ]; then
  cargo fmt --check
  cargo clippy --release --all-targets -- -D warnings
  cargo test --release
fi
cargo build --release --locked --offline
cp "$CARGO_TARGET_DIR/release/libtreeflat.so" "$out/libtreeflat-$(uname -m).so"
sha256sum "$out/libtreeflat-$(uname -m).so"
