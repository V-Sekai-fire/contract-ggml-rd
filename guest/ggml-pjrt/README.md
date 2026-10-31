# ggml-pjrt

A ggml backend whose device is any PJRT C API plugin. `ggml_backend_pjrt_attach(api)` makes the one device, and `ggml_backend_dev_init` gives a backend like any other.

## What it does

- A ggml buffer is a range of fake addresses. Each tensor in it owns one `PJRT_Buffer` once its bytes are on the device.
- `set_tensor` fills a host copy. `graph_compute` uploads it, once, and the second compute of the same graph uploads nothing.
- A partial write to bytes already on the device reads them back first. A range that leaves its tensor is refused.
- `graph_compute` writes the graph as a module (`pjrt-rd/program.h`), compiles it once per distinct module, executes it, and keeps each result as the bytes of its output tensor.
- A graph that writes into one of its own leaves is refused by `supports_op`, so ggml's scheduler places that node elsewhere.

## What it does not do

- It runs the module formats that the plugin accepts. With `pjrt-rd` that is the module `program.h` writes, not StableHLO.
- It keeps only tensors flagged `GGML_TENSOR_FLAG_OUTPUT` (or the last node). A graph whose intermediate values the caller reads back by tensor will read zeros.

## Test

`ggml_test` probe `pjrt`, graph `backend:<name>` or `backend:<name>:f32`: the same graph on ggml-rd directly and through this backend, every output bit for bit. Controls: a second compute must not upload, a write into a weight must move the output, writing the bytes back must restore it, and a reference output of all zeros fails the gate.
