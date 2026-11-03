// XLA CPU's f32 maximum, minimum and compare, for the StableHLO lowering (guest/pjrt-rd).
//
// ggml has no ops with XLA's semantics for these: XLA reads denormals as zero (it runs with DAZ and FTZ), returns
// a NaN from maximum and minimum if either input is one, and orders -0 below +0; a compare is IEEE after the
// flush. Each is GGML_OP_MAP_CUSTOM2 with ggml_rd_xla_binary, which is also the CPU reference, and a userdata
// that names the op. ggml-rd recognises the function and runs the kernel of that op
// (lean/Ggml/SlangCodegen/XlaCompare.lean). A compare writes 1.0 or 0.0.
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_rd_xla_binary_kind {
	GGML_RD_XLA_MAX = 0,
	GGML_RD_XLA_MIN,
	GGML_RD_XLA_GT,
	GGML_RD_XLA_GE,
	GGML_RD_XLA_LT,
	GGML_RD_XLA_LE,
	GGML_RD_XLA_EQ,
	GGML_RD_XLA_NE,
	GGML_RD_XLA_COUNT,
};

void ggml_rd_xla_binary(struct ggml_tensor *dst, const struct ggml_tensor *a, const struct ggml_tensor *b, int ith, int nth, void *userdata);

// a OP b, f32, a and b of the same shape.
struct ggml_tensor *ggml_rd_xla_binary_op(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, enum ggml_rd_xla_binary_kind kind);

// The same for one pair of values, for tests.
float ggml_rd_xla_binary_f32(enum ggml_rd_xla_binary_kind kind, float a, float b);

#ifdef __cplusplus
}
#endif
