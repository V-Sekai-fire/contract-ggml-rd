# StableHLO in pjrt-rd

`stablehlo.cpp` lowers one `func.func @main` of a StableHLO module to a ggml graph. Row-major shapes become ggml `ne` by reversing the dimensions. Constants carry their bytes in the `Program`, and arguments keep their declared order.

## Supported (23 ops)

`constant`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `rsqrt`, `dot_general`, `reduce`, `maximum`, `abs`, `floor`, `sign`, `slice`, `concatenate` and `pad`.

Element types are f32 and bf16, shapes are static and of rank 4 or less. bf16 element-wise ops run in f32 and round back, as XLA does. `dot_general` takes one contracting dimension and up to two batch dimensions per side, and at most one free dimension. `reduce` applies `add` (`ggml_sum_rows`) or `maximum` (`ggml_pool_1d`) over any set of dimensions, and its `init` must be the identity (0, or -inf). `maximum` is accepted only against a constant -inf, which is the no-op softmax writes. A hex literal is accepted only as an infinity. `rsqrt` is one over a square root. `slice` takes static starts, limits and strides (a stride on the last dimension is applied on a transposed copy, because a ggml view cannot step along its first dimension). `concatenate` takes any number of operands. `pad` takes edge padding with a constant zero and no interior padding. `sign` is a composite, because ggml-rd's `SGN` returns `-0` for a zero (ggml-cpu returns `+0`) and XLA returns the zero it was given: `s = sgn(x)`, `ss = s*s`, `c = s + sgn(1/x) * (1 - ss)`, `sign = c * ss`, which is exact for zeros of either sign, infinities and denormals (XLA flushes a denormal to a signed zero). A NaN gives 0, not NaN.

## Refused, by design

Everything else. The error is `UNIMPLEMENTED` for valid StableHLO outside the list (another op, another element type, rank 5, a dynamic dimension, a second function, hex or non-finite constants, returning a parameter unchanged) and `INVALID_ARGUMENT` for text that is malformed or inconsistent (an undefined operand, a result type that does not match the op, a shape mismatch). The probe holds all 14 refusals.

## Bit equality with XLA CPU

The probe compares every case with the output XLA's CPU backend gave for the same JAX module. 40 of 48 cases are bit-equal on the RTX 3090. Eight are not, and each carries its measured reason in `tools/gen_stablehlo_cases.py` (`XFAIL`). They must meet the functional bar (relative L2 of 1e-4 for f32, 1e-3 for bf16) and fail the probe if one becomes bit-equal, so the list only shrinks.

| Case | Why it is not bit-equal yet |
|---|---|
| exp, log | XLA CPU's results are approximations (9 and 18 of 128 elements differ from the correctly rounded value), and ggml-rd's differ from ggml-cpu's too (2168 and 2153 of 4096). Equality needs XLA's exact polynomial and its fused multiply-add use. |
| rsqrt | XLA CPU's rsqrt differs from one over a square root in f32 in 1244 of 4096 elements and from the correctly rounded value in 737. |
| reduce_sum_random, mean_last | XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds. `tools/reveal_sum_order.py` reveals it (the FPRev method): `(((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12)))))` for 8 rows of 16, and a different tree for 32, 48 and 64 elements. No closed-form order reproduces it; only a table of revealed trees per shape, tied to this CPU and XLA build, could. XLA lowers the reduce to one `vector.multi_reduction` (xla/backends/cpu/codegen/tiled), so the order is chosen after XLA's own code. ggml-rd sums 256 interleaved lanes. |
| dot_batched | XLA CPU sums this dot with a separate rounding per multiply and add. Its unbatched dot, and ggml-rd's mul_mat, use fused multiply-add. |
| softmax_f32, rms_norm_f32, mlp_f32 | each contains exp, a float sum or rsqrt from the rows above |

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
