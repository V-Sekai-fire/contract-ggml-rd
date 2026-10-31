// ggml-pjrt: a ggml backend whose device is any PJRT C API plugin.
//
//   buffer type  a ggml buffer is a fake address range (0x1000 + (index << 40), never
//                dereferenced). Every non-view tensor placed in it gets an entry, which owns
//                one PJRT_Buffer once its bytes are on the device. PJRT cannot write a range of a
//                device buffer, so set_tensor fills a host staging copy that graph_compute
//                uploads; a partial write to bytes already on the device reads the entry back
//                first. An address inside an entry (a view) resolves to that entry, and a range
//                that leaves it is refused, never copied short.
//   backend      graph_compute serializes the graph (pjrt-rd/program.h), compiles it once per
//                distinct module, executes it on the entry buffers, and keeps each result as the
//                entry of its output tensor. Only tensors flagged GGML_TENSOR_FLAG_OUTPUT (or the
//                last node, when none is flagged) are kept; a graph that writes into one of its
//                own leaves is refused.
#pragma once

#include <string>

#include "ggml-backend.h"
#include "pjrt_c_api.h"

ggml_backend_reg_t ggml_backend_pjrt_reg(void);

// Creates the client on `api` and makes the registry's one device. False, with
// ggml_backend_pjrt_last_error() set, if the plugin refuses.
bool ggml_backend_pjrt_attach(const PJRT_Api *api);

// Drops the compiled graphs and the client. Free every backend and buffer first.
void ggml_backend_pjrt_detach(void);

bool ggml_backend_is_pjrt(ggml_backend_t backend);

std::string ggml_backend_pjrt_last_error(void);

// Counters since attach, as one line.
std::string ggml_backend_pjrt_stats(void);
