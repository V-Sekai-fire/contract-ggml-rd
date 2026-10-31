# StableHLO in pjrt-rd

`stablehlo.cpp` lowers one `func.func @main` of a StableHLO module to a ggml graph. Row-major shapes become ggml `ne` by reversing the dimensions. Constants carry their bytes in the `Program`, and arguments keep their declared order.

## Supported (23 ops)

`constant`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `rsqrt`, `dot_general`, `reduce`, `maximum`, `abs`, `floor`, `sign`, `slice`, `concatenate` and `pad`.

Element types are f32 and bf16, shapes are static and of rank 4 or less. bf16 element-wise ops run in f32 and round back, as XLA does. `dot_general` takes one contracting dimension and up to two batch dimensions per side, and at most one free dimension. `reduce` applies `add` (`ggml_sum_rows`) or `maximum` (`ggml_pool_1d`) over any set of dimensions, and its `init` must be the identity (0, or -inf). `maximum` is accepted only against a constant -inf, which is the no-op softmax writes. A hex literal is accepted only as an infinity. `rsqrt` is one over a square root. `slice` takes static starts, limits and strides (a stride on the last dimension is applied on a transposed copy, because a ggml view cannot step along its first dimension). `concatenate` takes any number of operands. `pad` takes edge padding with a constant zero and no interior padding. `sign` is a composite, because ggml-rd's `SGN` returns `-0` for a zero (ggml-cpu returns `+0`) and XLA returns the zero it was given: `s = sgn(x)`, `ss = s*s`, `c = s + sgn(1/x) * (1 - ss)`, `sign = c * ss`, which is exact for zeros of either sign, infinities and denormals (XLA flushes a denormal to a signed zero). A NaN is found as the one input where `ss` and `sgn(1/x)` are both 0, and the result is multiplied by `1 + 0/(1 - ind)`, which is NaN there (0/0) and 1 elsewhere.

## Refused, by design

Everything else. The error is `UNIMPLEMENTED` for valid StableHLO outside the list (another op, another element type, rank 5, a dynamic dimension, a second function, hex or non-finite constants, returning a parameter unchanged) and `INVALID_ARGUMENT` for text that is malformed or inconsistent (an undefined operand, a result type that does not match the op, a shape mismatch). The probe holds all 14 refusals.

## Bit equality with XLA CPU

The probe compares every case with the output XLA's CPU backend gave for the same JAX module. 47 of 54 cases are bit-equal on the RTX 3090. Seven are not, and each carries its measured reason in `tools/gen_stablehlo_cases.py` (`XFAIL`). They must meet the functional bar (relative L2 of 1e-4 for f32, 1e-3 for bf16) and fail the probe if one becomes bit-equal, so the list only shrinks.

| Case | Why it is not bit-equal yet |
|---|---|
| rsqrt | XLA CPU's rsqrt is the CPU's hardware estimate (`rsqrtps`) plus Newton steps (`xla/codegen/intrinsic/rsqrt.cc`), and the source says it is not portable across CPUs. On the AMD Ryzen 7 3800X the estimate is a 2 x 4096-entry table keyed by the top 12 mantissa bits (no closed form: the best analytic fit is off by one unit in 24% of entries) followed by one fused Newton step. `tests/xla_rsqrt` holds a host model that equals XLA for all 2^32 inputs. Intel has a different table and two steps. A bit-equal GPU `rsqrt` needs a kernel with this CPU's table, and would match XLA on AMD only. |
| reduce_sum_random, mean_last | XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds. `tools/reveal_sum_order.py` reveals it (the FPRev method): `(((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12)))))` for 8 rows of 16, and a different tree for 32, 48 and 64 elements. No closed-form order reproduces it; only a table of revealed trees per shape, tied to this CPU and XLA build, could. XLA lowers the reduce to one `vector.multi_reduction` (xla/backends/cpu/codegen/tiled), so the order is chosen after XLA's own code. ggml-rd sums 256 interleaved lanes. |
| mlp_pre, mlp_f32, dot_batched | XLA CPU's dot (a YNNPACK fusion) accumulates in a shape-dependent tree. `tools/reveal_dot_order.py` shows that for 4x16 @ 16x8 it is the same tree as a float reduce of 16, and a sequential fused multiply-add chain matches 14 of 32 elements. The bias add after the dot is exact (dot then add equals XLA in 32 of 32), and so is the `exp` after it. |
| softmax_f32, rms_norm_f32 | each contains a float sum or rsqrt from the rows above. `softmax_pinned`, the same softmax with the sum order written out, is bit-equal. |

`exp` and `log` are bit-equal. XLA CPU does not call libm for them: `polynomial_approximations.cc` (revision d867969800d4) rewrites them to a Cephes-style polynomial, and `lean/Ggml/SlangCodegen/ExactExpLog.lean` is that code, with the fused multiply-adds LLVM chose (found by comparing a host model with XLA over all 2^32 inputs: 0 differ) and XLA's flush of subnormals done with the exponent bits. `tests/exact_exp_log` holds the host model and the sweep. It changes the `exp_f32` and `log_f32` kernels, which ggml's EXP and LOG use, to XLA's polynomial: within 2^-22.5 relative of the true value, a few ulp from libm's. The `exp` inside softmax, SiLU and flash attention is unchanged. `exp_stress` and `log_stress` run 4096 inputs each (random bit patterns, both clamp edges, the subnormal boundary, zeros, subnormals, infinities and a NaN) and are bit-equal on the GPU; with the old shaders they differ in 2045 and 1465 of 4096.

`divide` by a broadcast is lowered as XLA does: its simplifier rewrites `a / broadcast(b)` as `a * broadcast(1 / b)`, which rounds twice. For a divisor of 3 the two forms differ in the last place, and with the rewrite `divide_by_const3`, `divide_by_row` and `softmax_pinned` are bit-equal.

Integer-valued sums, `maximum` reductions (single and multi-dimension), bf16 sums and sums over a non-last or non-adjacent dimension are bit-equal.

## Pinning a sum's order

StableHLO leaves the order of a `reduce` unspecified, and XLA CPU's is an arbitrary tree (below). A program that needs a defined order has to write it, and write it so XLA keeps it. A chain of size-2 reduces does not hold: XLA's simplifier merges it into one `reduce-window` and the order changes. Explicit adds of slices do. `tree_sum` in `tools/gen_stablehlo_cases.py` adds lane `i` to lane `i + n/2` and halves until one is left, and it equals XLA CPU in every row for every shape tried (16 to 1024 elements, 1 to 64 rows). `pinned_sum_tree`, `pinned_mean` and `pinned_rms_scale` use it and are bit-equal; this is how to get a bit-equal float sum.

XLA also rewrites `a / sqrt(v)` into `a * rsqrt(v)`, so a full RMS norm meets the approximate `rsqrt` again; `rms_norm_f32` stays on the list below.

## Coverage

The goal is the whole of StableHLO, and every graph the ggml-rd probes run today (qwen, dit, sconv, the two Kimodo layers). Today it is 23 ops. The two Kimodo layers also need RoPE and attention, and their reductions need XLA's summation order to compare bit for bit.

## Regenerating the cases

```sh
pixi run --manifest-path <stablehlo-gen>/pixi.toml python tools/gen_stablehlo_cases.py guest/ggml_test/stablehlo_cases.inc
```

The environment is Python 3.12 with JAX from PyPI (conda-forge has no `jaxlib` for Windows). The header of the generated file names the JAX version.
