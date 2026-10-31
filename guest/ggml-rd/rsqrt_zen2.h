// rsqrt as XLA CPU computes it on an AMD Zen 2, for the StableHLO lowering (guest/pjrt-rd).
//
// ggml has no rsqrt op. This is GGML_OP_MAP_CUSTOM1 with a function that is also the CPU reference: the
// estimate comes from the table of RSQRTPS (the way Bochs emulates Intel's, cpu/sse_rcp.cc), then XLA's
// one Newton step with fused multiply-adds, and the estimate alone below the smallest normal float and for
// infinity (xla/codegen/intrinsic/rsqrt.cc). ggml-rd recognises the function and runs the kernel rsqrt_zen2_f32,
// which is the same computation. XLA's rsqrt is not portable across CPUs; on others it differs by an ulp or two.
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_rd_rsqrt_zen2(struct ggml_tensor *dst, const struct ggml_tensor *a, int ith, int nth, void *userdata);

// 1 / sqrt(a), f32, same shape.
struct ggml_tensor *ggml_rd_rsqrt_zen2_op(struct ggml_context *ctx, struct ggml_tensor *a);

// The same for one value, for tests.
float ggml_rd_rsqrt_zen2_f32(float x);

#ifdef __cplusplus
}
#endif
