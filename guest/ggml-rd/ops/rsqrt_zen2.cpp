// ggml-rd op: rsqrt as XLA CPU computes it on a Zen 2 (see ../rsqrt_zen2.h).
//
// The CPU reference below is the Bochs-style table emulation of RSQRTPS plus XLA's Newton step. The kernel is
// lean/Ggml/SlangCodegen/ExactRsqrt.lean (rsqrt_zen2_f32): the same estimate from the exact function and a 2-bit
// residual per entry, so it needs no table buffer. tests/xla_rsqrt/zen2_kernel_check.c compares the kernel's
// arithmetic with the hardware model for all 2^32 inputs, and the table with the instruction likewise.
#include "../rd_pack.h"
#include "../rsqrt_zen2.h"
#include "../rsqrt_zen2_table.h"

#include <cmath>
#include <cstring>

namespace {

using namespace ggml_rd;

float bits2f(uint32_t u) {
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

uint32_t f2bits(float f) {
	uint32_t u;
	std::memcpy(&u, &f, 4);
	return u;
}

// RSQRTPS on Zen 2: the table lookup, the way Bochs's approximate_rsqrt does it.
uint32_t estimate_bits(uint32_t op) {
	const uint32_t sign = op >> 31;
	const uint32_t exp = (op >> 23) & 0xff;
	const uint32_t fraction = op & 0x7fffff;
	if (exp == 0) {
		return (sign << 31) | 0x7f800000u; // zero or subnormal: a signed infinity
	}
	if (exp == 0xff) {
		if (fraction != 0) {
			return op | 0x00400000u; // NaN, quieted
		}
		return sign ? 0xffc00000u : 0u; // -inf: the default NaN; +inf: zero
	}
	if (sign) {
		return 0xffc00000u; // negative: the default NaN
	}
	const uint16_t *table = (exp & 1) ? zen2_rsqrt_table_odd : zen2_rsqrt_table_even;
	const int32_t sh = (int32_t(exp) - 0x7f) >> 1;
	return (uint32_t(0x7e - sh) << 23) | (uint32_t(table[fraction >> 11]) << 11);
}

// xla/codegen/intrinsic/rsqrt.cc on an AMD CPU without AVX-512: one Newton step, fused.
float rsqrt_one(float x) {
	const float y = bits2f(estimate_bits(f2bits(x)));
	if (x < bits2f(0x00800000u) || x == INFINITY) {
		return y;
	}
	const float r2 = x * y;
	const float half_r = y * -0.5f;
	const float h = std::fma(r2, y, -1.0f);
	return std::fma(half_r, h, y);
}

// The marker the packer looks for in the op's userdata.
const char kMarker = 'z';

struct CustomParams { // ggml_map_custom1_op_params in ggml.c
	void *fun;
	int n_tasks;
	void *userdata;
};

bool supports_rsqrt_zen2(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	CustomParams p;
	std::memcpy(&p, op->op_params, sizeof(p));
	if (p.fun != reinterpret_cast<void *>(&ggml_rd_rsqrt_zen2) || p.userdata != static_cast<const void *>(&kMarker)) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, op)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_rsqrt_zen2(Pack &p) {
	p.kernel = kernel_index("rsqrt_zen2_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

extern "C" {

float ggml_rd_rsqrt_zen2_f32(float x) { return rsqrt_one(x); }

void ggml_rd_rsqrt_zen2(struct ggml_tensor *dst, const struct ggml_tensor *a, int ith, int nth, void *) {
	(void)nth;
	if (ith != 0) {
		return;
	}
	for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
		for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
			for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
				for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
					const float *s = reinterpret_cast<const float *>(reinterpret_cast<const char *>(a->data) + i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
					float *d = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
					*d = rsqrt_one(*s);
				}
			}
		}
	}
}

struct ggml_tensor *ggml_rd_rsqrt_zen2_op(struct ggml_context *ctx, struct ggml_tensor *a) {
	return ggml_map_custom1(ctx, a, ggml_rd_rsqrt_zen2, 1, const_cast<char *>(&kMarker));
}

} // extern "C"

GGML_RD_OP(rsqrt_zen2_f32, GGML_OP_MAP_CUSTOM1, -1, supports_rsqrt_zen2, pack_rsqrt_zen2);
