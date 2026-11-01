# QuickForest: the standard build

One page, for QuickForest only. Decision of 31/10/2026: fat LTO + all threads (16 codegen units) + inline 3000, always; the PGO profile never on the rows that are timed.
The code follows this file; `tests/test_build_standard.py` fails when a default drifts from it.

## The rules

1. **Every build is fat LTO + 16 codegen units (all threads) + inline threshold 3000**: the evaluator library and the generated code, normal compilation included. What can run in parallel runs in parallel; one codegen unit is never a default, it is only built when a user passes `--codegen-units 1` by hand.
2. **The PGO build has the SAME shape.** The instrumented build has exactly the final build's flags; only `profile-generate` against `profile-use` differs.
3. **Profile data.** In normal use the profile is taken on ALL the data given (`--data`; at most `PROFILE_ROWS`, a seeded random pick, so a huge file stays bounded).
   Only when speed or accuracy is being tested or benchmarked, the profile is taken on rows DISJOINT from the rows that are timed or tested. A benchmark says which rows were which.
4. **CS-PGO** (context-sensitive PGO, the linker route) is part of the standard, with a switch to turn it off like the PGO one: `--cspgo on|off` (option `cspgo`), on by default wherever the build is profile-guided, fat-LTO, chunks and the toolchain has rust-lld (otherwise plain PGO with a notice); `off` is plain PGO; `on` refuses where it cannot be built. Measured on the six models (`bench/README-CSPGO.md`): within +-1 % of PGO at 1 thread on all six, noise at 2 threads (a 2-core machine), compile 2 to 4 times longer. The rule "CS-PGO only when it gains more than 1 % over PGO on most models" alone says no; it stays (decided 31/10) because the many-thread result is the one that matters.
5. **Published speed figures are the standard + PGO** (rule 2), profile and timed rows disjoint (rule 3).

## Where each rule lives

| rule | code | pinned by |
|---|---|---|
| 1, generated code | `scripts/backend_codegen.py`: `DEFAULT_COMPILE_PRESET = "1.1"`, `COMPILE_PRESETS["1.1"]` = `lto fat`, `codegen_units 16`, `inline_threshold 3000`, `pgo auto` | `test_build_standard.py::test_generated_code_default`, `test_compile_settings.py::test_the_default_is_the_1_1_preset_behind_one_constant` |
| 1, evaluator | `rust/Cargo.toml` `[profile.release]` (`lto = "fat"`, `codegen-units = 16`) and `rust/.cargo/config.toml` (`-C llvm-args=-inline-threshold=3000`); `rust/build.sh` unsets RUSTFLAGS so the file stays in force | `test_build_standard.py::test_evaluator_profile` |
| 2 | `chunk_flags` / `driver_flags` give the instrumented and the final build the same flags | `test_compile_settings.py::test_the_instrumented_and_the_final_build_share_the_shape_in_every_combination` |
| 3 | `compile.py: profile_rows` (all rows up to `PROFILE_ROWS`); `bench/` and `compare.py prepare` take the profile from rows outside the timed ones (`x-pgo.npy` vs `x-speed.npy`) | `test_build_standard.py::test_benchmark_profile_rows_are_disjoint_from_the_timed_rows` |
| 4 | `scripts/backend_codegen.py`: `COMPILE_PRESETS[...]["cspgo"] = "auto"`, `cspgo_unavailable`, `cspgo_stages`; `--cspgo on|off` | `test_build_standard.py::test_cspgo_default`, `test_cspgo_switch.py` (settings, plan with the toolchain faked, a real build both ways) |

Fat LTO only crosses chunks when the whole forest is ONE crate (`--unit single`); on separate chunk crates it is a no-op (objects byte-identical, rustc 1.99),
so the chunks build asks it of the driver's link only. `DEFAULT_COMPILE_UNIT` chooses between the two; the measured figures for both are in `bench/README-CSPGO.md`.
