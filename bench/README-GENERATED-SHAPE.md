# Why generated code builds in the `fast` shape

Generated code (`codegen-block`, `codegen-pgo`) is built with `-C opt-level=3 -C lto=off -C codegen-units=16`, and the profile-guided pass builds its
instrumented binary and its final binary in that same shape. `--shape top` builds one codegen unit per chunk instead. This note is the measurement
behind that default, with what it does not show.

## What was measured

A text-feature preparation crate (a library that turns document text into numeric features; not a forest and not generated code), built for
x86-64-v2 at opt-level 3, on one shared laptop, timed in 28 interleaved rounds of the same documents (3 repetitions of every process). Four ways to
build it:

| | profile gathered with | final build | speed against no profile in the top-tier shape | build process time |
|---|---|---|---|---|
| no profile, top-tier shape | - | top-tier | baseline | 5.6 s |
| no profile, fast shape | - | fast | +2.6 % (+0.7 to +4.2) | 1.5 s |
| P1 | top-tier | top-tier | +10.5 % (+8.3 to +11.6) | 12.4 s |
| P2 | fast | top-tier | +0.5 % (-0.8 to +3.4): the profile matched 0 of 9 hot functions | 8.1 s |
| P3 | fast | fast | **+14.6 %** (+11.8 to +17.1) | 4.1 s (-67 % against P1) |

(Intervals are 95 % confidence intervals of the median.) P3 against P1 is +4.0 % (+1.0 to +6.2), and P3 against the fast shape without a profile
is +9.8 % (+8.6 to +14.0). Top-tier shape here means fat link-time optimisation and one codegen unit; fast shape means no link-time optimisation and
16 codegen units. Every final binary produced the same output digests.

The run could not resolve differences below about 3 points: the same binary timed twice differed by +0.2 % (-2.2 to +3.0). So the profile against no
profile (P1, P3) and the failure of P2 are resolved, and P3 against P1 (+4.0 %) is the marginal figure.

## What it means for this tool

- The profile-guided pass of `codegen-pgo` runs in one shape end to end because P2 shows what happens otherwise: rustc applies a profile that
  does not belong to its code without a word, and the function that lost its counts just runs unoptimised. `codegen-pgo` fails in one line when a
  hot function (the fewest that hold 90 % of the profile's counts) lost its profile, and counts the cold ones (`docs/REFERENCE.md`, the stale-profile gate), and its instrumented and final builds share their flags.
- The generated code of this tool is not that crate: its chunks are separate crates joined at link time, so `top` here is one codegen unit per
  chunk and not fat link-time optimisation, and a chunk's trees are written into modules so that 16 codegen units are real units. How much the
  `fast` shape gains or loses on a forest's generated code is in the paired table of `bench/README-NUMBERS.md` (the benchmark's 10 % forest: `top` ran at 1.003 of `fast` on one thread and 1.005 on 8, inside the noise of the method); answers are bit-identical in both shapes
  (`tests/test_codegen_target.py`).
- Compiling in 16 units lets each `rustc` use several cores. It can also raise its peak memory, which is why the build plans its jobs against the
  free RAM (`docs/REFERENCE.md`).
- Building for the machine's own CPU added +0.6 % on that crate, inside the noise, so the default stays portable (`--native` is the opt-in).

## Not shown

One crate, one machine, one compiler version; the harness is not part of this repository, so these figures are not reproducible from it.
