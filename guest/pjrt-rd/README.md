# pjrt-rd

A PJRT C API plugin whose one device is ggml-rd. It is linked into a sandbox guest, so `GetPjrtApi()` is a function call and not a `dlopen`.

## What it is for

It lets a PJRT client (the ggml-pjrt backend, or the `pjrt` probe in `guest/ggml_test`) run a ggml graph through the PJRT C API on the engine's RenderingDevice, with the same kernels ggml-rd already checks.

## What it does

- One `PJRT_Buffer` is one ggml-rd buffer. `BufferFromHostBuffer`, `ToHostBuffer` and `CopyRawToHost` move bytes in chunks of at most 16 MiB. A range read that leaves the buffer fails with `OUT_OF_RANGE` and writes nothing.
- `Client_Compile` takes format `mlir` and accepts two kinds of module. A module in the ggml-carrying form that `program.h` writes (each ggml tensor is one op, `ggml.arg` for an argument and a `stablehlo.custom_call` to `ggml.<OP>` for the rest) runs unchanged. A genuine StableHLO module is lowered by `stablehlo.cpp` (see STABLEHLO.md). Valid StableHLO the lowering does not cover is refused with `UNIMPLEMENTED`, and malformed or inconsistent text with `INVALID_ARGUMENT`.
- `LoadedExecutable_Execute` rebuilds the graph on the argument buffers and runs it on ggml-rd. Results are new buffers.
- Every other entry of `PJRT_Api` returns `UNIMPLEMENTED`. `gen_stubs.py` writes `pjrt_rd_stubs.inc` from the header.
- `pjrt_c_api.h` is OpenXLA's, unchanged, at the commit in `PJRT_C_API_SOURCE`.

## Test

The `pjrt` probe of `ggml_test.elf` drives only the C API: `pjrt buffers`, `pjrt kimodo_denoiser` or `pjrt kimodo_text` (with `:f32` for the f32 arm), `pjrt backend:<graph>` and `pjrt stablehlo:all`. The graph probes compare each result bit for bit with the same graph run directly on ggml-rd. The StableHLO probe compares with the output XLA's CPU backend gives for the same module. Each check has a control that must fail.
