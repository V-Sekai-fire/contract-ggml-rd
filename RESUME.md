# RESUME: where the pjrt-rd / ggml-pjrt / StableHLO work stands

Written 2026-10-31 from the working session, for whoever (person or agent) picks this up. Everything below was measured on
the Windows desk unless it says otherwise: an RTX 3090 and an AMD Ryzen 7 3800X (Zen 2, AVX2 and FMA, no AVX-512), Godot
`4.8.dev.double` (`v20260930-double.1`) with the local RenderingDevice, slangc 2026.20, clang/lld 22.1, jax and jaxlib
0.12.0.1, python 3.12. `main` is at `c816784` (PR #28) when this was written.

## 1. The goal

1. Build the `pjrt-rd` plugin (a PJRT C API plugin over `ggml-rd` / `rd_compute`) and a `ggml-pjrt` backend, and run Kimodo
   end to end on the 3090 through Godot's local RenderingDevice.
2. Cover 100% of StableHLO, with **bit equality with XLA CPU** except where that is provably impossible (fact #659).
3. Cover 100% of our existing ggml users: qwen, dit, sconv, kimodo_denoiser and kimodo_text (fact #661).

The equality bar: bit equality is the target. The rel-L2 bar (<= 1e-3 native, <= 1e-4 for f32) applies only where bit
equality is shown impossible. Every fallback records its measured reason in the XFAIL table in
`guest/pjrt-rd/tools/gen_stablehlo_cases.py`. The table may only shrink, and the gate fails if an XFAIL case turns out to be
bit-equal ("NOT NEEDED: remove it from XFAIL"). Do not claim "impossible" without a proof or a measurement; say what was
measured.

## 2. Working rules (from the user)

- Tools come from pixi. Documented exceptions: Godot, slangc, and Lean via elan. (`AGENTS.md`, `check_no_auto.py` and
  `check_commit_style.py` live in the weftspun workspace repo, not in this one.)
- An unmet precondition is a FAIL, not a silent skip (working-agreements rule 3).
- Configure through the host's own mechanism, not environment variables (`AGENTS.md`).
- Every gate needs a negative control. A zero-vs-zero match is not evidence; a gate must fail on an all-zero reference.
- Do not touch shared checkouts. Work in scratch clones and keep shared repos clean.
- Read the guest's own `RESULT:` line, not the job state.
- Ask before any push. **Do not run `gh` at all** (rescinded, fact #662). Push with plain `git` and give the user a compare
  URL plus a one-sentence PR body; the user opens and merges the PR. CI status cannot be read without `gh`, so if a check
  fails the user pastes the job log.
- `main` refuses direct pushes (repository rule violations). Branch, push, compare URL.
- New code has no `auto` (`check_no_auto.py`). Commit style: `check_commit_style.py --base ...`, sentence case.
- Reply style: calibrated words with probabilities for unverified claims (fact #218, #42); PR bodies are one sentence.
- Float sums: "whatever gets us bit equality. you choose."

## 3. What exists on `main`

PRs merged on this repo so far: #14 through #28 (all fifteen; #1 to #13 predate this work). In `transport-meshing-pen`
(the build tree) PRs #108 and #109 merged.

### NOT merged: one line in `transport-meshing-pen`

`transport-meshing-pen` branch `dev/pjrt-rd-stablehlo` (commit `22047c7`, pushed, one line in `CMakeLists.txt`: "Build
stablehlo.cpp into the pjrt_rd library") is **not on that repo's `main`**. Its PR (#110) failed once (cause inferred as
timing, never confirmed because the CI log was not readable) and was not merged. Until it lands, a fresh meshing-pen checkout
builds `pjrt_rd` without `stablehlo.cpp`. Compare: https://github.com/V-Sekai-fire/transport-meshing-pen/compare/main...dev/pjrt-rd-stablehlo
(suggested body: "Builds stablehlo.cpp into the pjrt_rd library."). Everything measured in this file used that branch's build.

| Piece | Where |
|---|---|
| PJRT plugin | `guest/pjrt-rd/` (`pjrt_rd.cpp`, `program.cpp`, `stablehlo.{h,cpp}`, `STABLEHLO.md`, `tools/`) |
| ggml backend over PJRT | `guest/ggml-pjrt/` |
| ggml-rd ops and the CPU reference ops | `guest/ggml-rd/` (`ops/*.cpp`, `rsqrt_zen2.h`, `rsqrt_zen2_table.h`) |
| Probe and cases | `guest/ggml_test/probe_pjrt.cpp`, `stablehlo_cases.inc` (generated) |
| Lean kernel generators | `lean/Ggml/SlangCodegen/` (`ExactDivSqrt`, `ExactExpLog`, `ExactRsqrt`, `ExactSinCos`, `UnaryMath`, ...) |
| Generated kernels | `kernels/ggml/slang/*.slang`, `kernels/ggml/cpp/*_emit.cpp`, `GgmlKernelTable.inc` |
| Host proofs and measurements | `tests/exact_div_sqrt`, `tests/exact_exp_log`, `tests/xla_rsqrt`, `tests/xla_sincos` |

### StableHLO ops lowered (`guest/pjrt-rd/stablehlo.cpp`)

`constant`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`,
`exponential`, `log`, `sqrt`, `rsqrt`, `sine`, `cosine`, `dot_general` (one contracting dimension, up to two batch
dimensions), `reduce` (add, maximum), `maximum`, `abs`, `floor`, `sign`, `slice`, `concatenate`, `pad` (zero value, no
interior padding). Element types f32 and bf16, static shapes, rank <= 4. bf16 element-wise ops run in f32 and round back.
Unsupported-but-valid ops return `UNIMPLEMENTED`, malformed ones `INVALID_ARGUMENT`. The refusal tests (22 of them)
must hold; the unsupported-op example is now `stablehlo.tan`.

### Current probe state

58 cases, 51 bit-equal, 7 expected-not-bit-equal, 58 controls held. Eleven gates pass:

```
--probe=pjrt --graph=stablehlo:all     --probe=pjrt --graph=buffers
--probe=pjrt --graph=kimodo_denoiser   --probe=pjrt --graph=kimodo_text
--probe=pjrt --graph=backend:kimodo_denoiser   --probe=pjrt --graph=backend:kimodo_text
--graph=kimodo_denoiser  --graph=kimodo_text  --graph=qwen  --graph=dit:8  --graph=sconv
```

Case history: 24 (18 bit-equal) -> 24 (20) -> 34 (26) -> 44 (35) -> 48 (40) -> 54 (47) -> 55 (48) -> 57 (50) -> 58 (51).
The earlier baseline was `kimodo_denoiser` / `kimodo_text` FAIL with `drop_control=0/48`; both now pass.

The 7 expected-not-bit-equal entries (XFAIL): `reduce_sum_random`, `mean_last`, `softmax_f32`, `rms_norm_f32`,
`dot_batched`, `mlp_f32`, `mlp_pre`. All are float-sum or dot-order cases (section 5).

## 4. How each op reached bit equality (and what the reference really is)

The reference is always XLA CPU run by JAX on the same machine. The procedure that worked each time: read the source of
truth (the XLA source at the pinned revision, or the disassembly of the library XLA calls), write a host model, sweep all
2^32 inputs against the reference, port to a Lean-emitted Slang kernel, run a stress case on the GPU, and keep a negative
control (the old kernel, or a wrong kernel, must fail the same case).

XLA pin: `d867969800d4a8fd045505eabc65e1e66f30794f`.

| Op | What XLA CPU does | How it is matched | Evidence |
|---|---|---|---|
| `divide`, `sqrt` | IEEE correctly rounded | exact kernels (`ExactDivSqrt.lean`); host proofs over all 2^32 patterns | `tests/exact_div_sqrt` |
| `divide` by broadcast | rewritten to `a * broadcast(1/b)` | lowering does the same (`Val.pre`) | `softmax_pinned`, `divide_by_row` |
| `exp`, `log` | XLA's own f32 polynomials (`polynomial_approximations.cc`, `GenerateVF32Exp/Log`), subnormals flushed | `ExactExpLog.lean`; model `xla_expl.c`; 0 differences over 2^32 | `tests/exact_exp_log` |
| `rsqrt` | CPU estimate (`rsqrtps`) + one fused Newton step (AMD, no AVX-512); NOT portable | Zen 2 table (2 x 4096 entries) emulated Bochs-style: `GGML_OP_MAP_CUSTOM1` + `ExactRsqrt.lean` (exact function + 2-bit residual per entry, 512 words in an `if` tree) | `tests/xla_rsqrt`, `rsqrt_stress` |
| `sine`, `cosine` | the C runtime's `sinf` / `cosf` (UCRT here), FMA path, double precision | `ucrt_libm.c` translation of the disassembly, equal to the runtime for all 2^32 inputs; `ExactSinCos.lean` kernel (double + `uint64_t`, Payne-Hanek table as 22 words) | `tests/xla_sincos`, `sin_stress`, `cos_stress` |
| `sign` | returns the zero it was given; ggml-rd `SGN` returns -0, ggml-cpu +0 | composite op from exact arithmetic, NaN-safe | `sign_special` |
| `abs`, `floor`, `slice`, `concatenate`, `pad` | exact | ggml ops; a stride on the last dimension is applied on a transposed copy | 44-case set |
| float sums | an irregular, shape-dependent tree (FPRev) | **pin the order**: write the sum as explicit slice adds (`tree_sum`: `a = a[..., :h] + a[..., h:]`) | `pinned_sum_tree`, `pinned_mean`, `pinned_rms_scale` |
| `rope_neox` | sin, cos, slice, concat, mul, add, sub | works with the existing lowering (frequency table passed as an input) | `rope_neox`, 0/512 differ |

Notes that matter:

- **rsqrt** is machine-specific. The estimate depends only on exponent parity (fields 127/128) and the top 12 mantissa bits.
  No closed form was found: tried exact 1/sqrt at group edges, floored polynomials of degree 1-3 over 1-64 segments, linear
  interpolation between nodes, an LP over continuous nodes, tangent lines, a plausible-witness-dag search over 4096
  hypotheses (provably none; closest is wrong in 1,959 of 8,192). Intel's table has a closed form (Bochs `sse_rcp.cc`:
  `round(1/sqrt(midpoint))` to 13 bits, 0 mismatches of 1024) but AMD's is a 12-bit-index, 4096-entry table that does not
  follow it. The table as a decision DAG has 186 and 167 distinct nodes (`tests/xla_rsqrt/table_dag.py`). Intel would need a
  different table and two Newton steps.
- **sin / cos / (powf)** are libm calls, so they are platform-specific: matching needs UCRT's algorithm; another platform's
  libm gives other bits, and a CPU without FMA takes UCRT's other path. XLA's `XLA_FLAGS` do not change any of this.
- ggml's own `EXP`, `LOG`, `SIN`, `COS` kernels were replaced by the XLA / UCRT versions (within a few ulp of libm). The
  `exp` inside softmax, SiLU and flash attention is unchanged. qwen, dit, sconv, kimodo still pass their oracles.

## 5. Open problems (in order of how much they block)

1. **XLA's dot order.** `dot_batched`, `mlp_f32`, `mlp_pre` differ by up to 24 ulp (rel-L2 meets the bar). Measured with
   `guest/pjrt-rd/tools/dot_trees.py` (FPRev over the contracted dimension K):
   - with N > 1 the tree depends on K only (K=16 is identical for M x N of 1x8, 4x8, 8x8, 16x16, 3x5);
   - with N = 1 it is a lane-wise tree that pairs element i with i+8;
   - the N > 1 trees are chains inside blocks joined unevenly: K=2..4 left-to-right chain; K=9 three chains of three;
     K=12 a chain of six + chains of three; K=16 blocks 0..5, 6..10, 11..15; K=24, 32 bigger and uneven. No closed rule.
   Next measurement: large K (up to a few thousand) to see whether the trees become regular (fixed blocks). If a rule
   exists, a bit-equal `dot_general` is feasible for any K (built from gathers/`get_rows` and adds); if not, a per-K table.
2. **Unpinned float sums from `reduce`.** XLA's tree is irregular and shape-dependent; flags do not control it, and a chain of
   size-2 reduces gets merged into one `reduce-window`. A program that writes the sum as slice adds is bit-equal.
3. **`power`.** XLA CPU's `power` equals UCRT's `powf` (0 of 600,000 differ; `(float)pow(double)` differs in 53;
   `expf(b*logf(a))` differs in 393,319). Not translated. The FMA path is at ucrtbase.dll RVA 0xA9F20 (branch at +0x7b6, a
   ~400-instruction routine with log and exp tables and many special cases). It cannot be swept over 2^64 pairs, so the check
   would be hundreds of millions of random and structured pairs. Needed for a RoPE whose frequency table is computed in the
   program (`10000 ** (arange/d)`).
4. **Attention.** A causal attention block needs `iota`, `compare`, `select`, `convert`, and the `i32` / `i1` types (currently
   refused: f32 and bf16 only), plus `dot_general` (already lowered) and softmax. Plan: fold `iota` + `compare GE` to a
   compile-time 0/-inf bias, lower `select(mask, x, -inf)` as `x + bias` (the one difference from XLA is the sign of zero:
   `-0 + 0 = +0`; measure and record it), then run attention at the rel-L2 bar until the dot is solved.
5. **Not yet measured:** `tanh`, `logistic`, `erf`, `expm1`, `log1p`, `tan` (XLA CPU routes several through YNNPACK / its own
   intrinsics); `reduce-window`; integer ops; `while` / control flow; gather / scatter; sort; conv. Use the StableHLO spec
   (https://openxla.org/stablehlo/spec) for the op list.
6. **Other platforms.** Everything above is bit-equal to XLA on this Windows / Zen 2 desk. A Linux or Intel reference needs its
   own rsqrt table, glibc `sinf` / `cosf` / `powf`, and re-measured dot trees.

## 6. Build and run (Windows desk recipe)

Kernel generation (after editing Lean): put `~/.elan/bin` on `PATH`, then

```bash
unset MSYS_NO_PATHCONV MSYS2_ARG_CONV_EXCL
export BUILD_DIR=$S/weft/gb SLANGC=/c/Users/ernest.lee/tools/slang-2026.20/bin/slangc.exe
bash kernels/ggml/gen.sh --update      # slow; run in the background; then restore every .slang you did not mean to change
```

Guest build (from `weft/1-transport/meshing-pen`, about 10 minutes now that the sin/cos kernels are big; run it in the
background):

```bash
export SLANGC=... WEFT_ROOT=$S/weft RISCV64_SYSROOT=$WEFT_ROOT/5-repository/riscv64-sysroot BUILD_TARGETS=ggml_test BUILD_JOBS=8
pixi run --manifest-path $S/pjrt-rd-env/pixi.toml bash build.sh
git -C weft/2-contract/ggml-rd checkout -q -- kernels/ggml/cpp kernels/ggml/GgmlKernelTable.inc   # build.sh rewrites these
cp ggml_test.elf $S/pen-build/
```

Run (from `pen-build`):

```bash
godot.windows.editor.double.x86_64.llvm.console.exe --path . --xr-mode off --rendering-driver vulkan \
  --script tools/run_ggml_graph.gd -- --probe=pjrt --graph=stablehlo:all       # one case: --graph=stablehlo:rope_neox
```

Regenerate cases (needs the JAX env): `pixi run --manifest-path $S/stablehlo-gen/pixi.toml python guest/pjrt-rd/tools/gen_stablehlo_cases.py guest/ggml_test/stablehlo_cases.inc`.
The JAX env is python 3.12 with `jax >=0.12.0.1,<0.13`. The full recipe is in the skill `godot-sandbox-build-workflow`
(`references/fresh-clone-weft-root.md`, `references/pjrt-rd-plugin.md`, `references/windows-desk-pixi-build.md`).

Host proofs (clang from the pixi env, `-O2 -mavx2 -mfma -ffp-contract=off`): see each `tests/*/README.md`. The 2^32 sweeps
take 1 to 4 minutes each.

## 7. Hazards

- Every `build.sh` run regenerates `kernels/ggml/cpp/*_emit.cpp` and `GgmlKernelTable.inc`. Restore them before committing,
  keeping only the files you changed on purpose. My slangc writes a different prelude, so the committed `*_emit.cpp` for
  div, sqrt, exp, log, rsqrt_zen2, sin and cos were not regenerated byte-for-byte upstream; CI may flag that.
- `git stash -u` and rebase churn CRLF on the emit files; restore with `git checkout -- kernels/ggml/cpp kernels/ggml/GgmlKernelTable.inc`.
- Edit helpers must preserve CRLF (`newline=""`).
- A case selector needs the `stablehlo:` prefix. A tool-call that exceeds the foreground limit (the Lean `gen.sh`, the guest
  build) should be started in the background.
- The `oracle:<op>` probe hangs when two operands are live.
- The `ggml` checkout is `main` with no recorded pin.
- Godot prints `Error loading extension: 'res://addons/cineform/cineform.gdextension'`; it is unrelated.
- Windows: DLLs need `__declspec(dllexport)`; use `tail -n 2` (not `tail -2`) with several files; MSYS path conversion is off, so
  pass `C:/...` paths to native tools.
- The Windows desk's tree was lost once (`C:/smoke-water`) and rebuilt under `scratch/weft/` from `V-Sekai-fire` clones.
  `contract-ggml-stablehlo` is reachable but was discarded; do not resurrect it.
- Credentials: none are in this repo. A one-off token existed and is gone; the OpenBao creds dir on the desk exists but the nodes
  were unreachable. Never type secrets with the browser tool.

## 8. Where the work physically lives (read this before cleaning anything)

- The session's tree is under Hermes's scratch directory `C:/Users/ernest.lee/AppData/Local/hermes/cache/scratch/`, which
  Hermes prunes after 24 idle hours. Pushed branches are the durable copy. Nothing unpushed matters except: untracked `build*.log`
  files in `weft/1-transport/meshing-pen` (disposable) and the exploratory host scripts in `scratch/exact/` (about 100 files:
  the formula-hunt scripts `rsqrt_*.py`, `patch_*.py` edit helpers, `pinned_check*.py`, `sign_*.py`). The ones that mattered were
  copied into `tests/`; the rest can be re-derived from section 4.
- Also under scratch: `weft/` (the tree rebuilt from `V-Sekai-fire` clones), `pjrt-rd-env/` and `stablehlo-gen/` (pixi projects),
  `pen-build/` (the Godot run directory with the logs `sh*.log`, `r9-*` ... `r18-*`), `pwd-lib/` (a clone of
  `plausible-witness-dag`), `rsqrt-witness/`.
- The skill `godot-sandbox-build-workflow` (in `~/.hermes/skills`) holds the build recipe and `references/pjrt-rd-plugin.md`.
- Sessions: the live Hermes session was `20261031_120418_369657`; recover older detail with `session_search` on that id.

## 9. Suggested next steps

0. Open the `transport-meshing-pen` PR above (the user opens and merges it).
1. Measure XLA's dot trees for large K and decide: rule, table, or stay at the rel-L2 bar (section 5.1).
2. `iota` / `compare` / `select` mask pieces, then a causal-attention case at the rel-L2 bar.
3. Translate UCRT `powf` and lower `power` and f32 `iota`, so computed RoPE tables are bit-equal.
4. Measure `tanh`, `logistic`, `erf`, `expm1`, `log1p`, `tan`, then `reduce-window`, `while`, gather/scatter, sort, conv.
5. Re-run the five ggml users and Kimodo end to end through the PJRT path as ops land, and keep the XFAIL table shrinking.

## 10. Related facts and references

- Mooncake PR #2733 was the model for the ggml mapping (`jax` -> StableHLO -> ggml graph + `Program`).
- RFD 2335 was deleted upstream by `be28a1f1`; RFD 2294 says agent knowledge lives in RFDs, not desk memory.
- `pjrt_c_api.h` is PJRT API 0.117. The plausible-witness-dag library is `V-Sekai-fire/plausible-witness-dag`
  (HEAD `b150a7ff29e51b65074bbe71b63d6954eb80caa5` when used), already a dependency of `csr_falsify` and `lasso`.
- Bochs `cpu/sse_rcp.cc` (Stanislav Shwartsman) documents Intel's `rcpps` / `rsqrtps` tables.
- XNNPACK's `src/f32-vsin/rational-5-4.c.in` is YNNPACK's own sin/cos (a 5/4 rational); XLA CPU's jitted `sine` does not use it
  (58% of inputs differ).
- Facts in the holographic store: #656 (pjrt-rd plan), #659, #661, #662 (no `gh`), #663 (direct commit to `main` is infeasible).
