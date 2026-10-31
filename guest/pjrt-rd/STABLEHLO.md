# StableHLO in pjrt-rd

`stablehlo.cpp` lowers one `func.func @main` of a StableHLO module to a ggml graph. Row-major shapes become ggml `ne` by reversing the dimensions. Constants carry their bytes in the `Program`, and arguments keep their declared order.

## Supported (17 ops)

`constant`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `rsqrt`, `dot_general`, `reduce` and `maximum`.

Element types are f32 and bf16, shapes are static and of rank 4 or less. bf16 element-wise ops run in f32 and round back, as XLA does. `dot_general` takes one contracting dimension and up to two batch dimensions per side, and at most one free dimension. `reduce` applies `add` (`ggml_sum_rows`) or `maximum` (`ggml_pool_1d`) over any set of dimensions, and its `init` must be the identity (0, or -inf). `maximum` is accepted only against a constant -inf, which is the no-op softmax writes. A hex literal is accepted only as an infinity. `rsqrt` is one over a square root.

## Refused, by design

Everything else. The error is `UNIMPLEMENTED` for valid StableHLO outside the list (another op, another element type, rank 5, a dynamic dimension, a second function, hex or non-finite constants, returning a parameter unchanged) and `INVALID_ARGUMENT` for text that is malformed or inconsistent (an undefined operand, a result type that does not match the op, a shape mismatch). The probe holds all 14 refusals.

## Bit equality with XLA CPU

The probe compares every case with the output XLA's CPU backend gave for the same JAX module. 26 of 34 cases are bit-equal on the RTX 3090. Eight are not, and each carries its measured reason in `tools/gen_stablehlo_cases.py` (`XFAIL`). They must meet the functional bar (relative L2 of 1e-4 for f32, 1e-3 for bf16) and fail the probe if one becomes bit-equal, so the list only shrinks.

| Case | Why it is not bit-equal yet |
|---|---|
| exp, log | XLA CPU's results are approximations (9 and 18 of 128 elements differ from the correctly rounded value), and ggml-rd's differ from ggml-cpu's too (2168 and 2153 of 4096). Equality needs XLA's exact polynomial and its fused multiply-add use. |
| rsqrt | XLA CPU's rsqrt differs from one over a square root in f32 in 1244 of 4096 elements and from the correctly rounded value in 737. |
| reduce_sum_random, mean_last | XLA CPU's float summation order depends on the shape. For one 64x64 input it is 16 interleaved lanes combined by halving; for rows of 16 or 32 it is sequential; for 4 to 64 rows of 48 to 1024 elements none of nine orders tried reproduces it. ggml-rd sums 256 interleaved lanes. This is not shown to be impossible: it needs XLA CPU's reduction emitter reproduced. |
| dot_batched | XLA CPU sums this dot with a separate rounding per multiply and add. Its unbatched dot, and ggml-rd's mul_mat, use fused multiply-add. |
| softmax_f32, rms_norm_f32, mlp_f32 | each contains exp, a float sum or rsqrt from the rows above |

Integer-valued sums, `maximum` reductions (single and multi-dimension), bf16 sums and sums over a non-last or non-adjacent dimension are bit-equal.

## Coverage

The goal is the whole of StableHLO, and every graph the ggml-rd probes run today (qwen, dit, sconv, the two Kimodo layers). Today it is 17 ops. The two Kimodo layers also need RoPE and attention, and their reductions need XLA's summation order to compare bit for bit.

## Regenerating the cases

```sh
pixi run --manifest-path <stablehlo-gen>/pixi.toml python tools/gen_stablehlo_cases.py guest/ggml_test/stablehlo_cases.inc
```

The environment is Python 3.12 with JAX from PyPI (conda-forge has no `jaxlib` for Windows). The header of the generated file names the JAX version.
