# StableHLO in pjrt-rd

`stablehlo.cpp` lowers one `func.func @main` of a StableHLO module to a ggml graph. Row-major shapes become ggml `ne` by reversing the dimensions. Constants carry their bytes in the `Program`, and arguments keep their declared order.

## Supported (14 ops)

`constant`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `dot_general`.

Element types are f32 and bf16, shapes are static and of rank 4 or less. bf16 element-wise ops run in f32 and round back, as XLA does. `dot_general` takes one contracting dimension and up to two batch dimensions per side, and at most one free dimension.

## Refused, by design

Everything else. The error is `UNIMPLEMENTED` for valid StableHLO outside the list (another op, another element type, rank 5, a dynamic dimension, a second function, hex or non-finite constants, returning a parameter unchanged) and `INVALID_ARGUMENT` for text that is malformed or inconsistent (an undefined operand, a result type that does not match the op, a shape mismatch). The probe holds all 14 refusals.

## Bit equality with XLA CPU

The probe compares every case with the output XLA's CPU backend gave for the same JAX module. 18 of 24 cases are bit-equal. Six are not, and each carries its measured reason in `tools/gen_stablehlo_cases.py` (`XFAIL`). They must meet the functional bar (relative L2 of 1e-4 for f32, 1e-3 for bf16) and fail the probe if one becomes bit-equal, so the list only shrinks.

| Case | Why it is not bit-equal yet |
|---|---|
| divide, sqrt | XLA CPU returns the correctly rounded value. ggml-rd's DIV and SQRT kernels are 1 ulp off in a quarter of the elements, so they need correctly rounded kernels. |
| exp, log | XLA CPU's results are approximations (9 and 18 of 128 elements differ from the correctly rounded value). Equality needs XLA's exact polynomial and its fused multiply-add use. |
| dot_batched | XLA CPU sums this dot with a separate rounding per multiply and add. Its unbatched dot, and ggml-rd's mul_mat, use fused multiply-add. |
| mlp_f32 | contains exp |

## Coverage

The goal is the whole of StableHLO, and every graph the ggml-rd probes run today (qwen, dit, sconv, the two Kimodo layers). Today it is 14 ops. The two Kimodo layers also need reductions, a softmax, RMS norm, RoPE and attention, none of which the lowering handles yet.

## Regenerating the cases

```sh
pixi run --manifest-path <stablehlo-gen>/pixi.toml python tools/gen_stablehlo_cases.py guest/ggml_test/stablehlo_cases.inc
```

The environment is Python 3.12 with JAX from PyPI (conda-forge has no `jaxlib` for Windows). The header of the generated file names the JAX version.
