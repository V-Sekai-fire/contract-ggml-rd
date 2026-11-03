// ggml-rd op: XLA CPU's f32 maximum, minimum and the six compares, as GGML_OP_MAP_CUSTOM2 (xla_compare.h).
//
// The kernels are lean/Ggml/SlangCodegen/XlaCompare.lean: Binary's shape (one thread per dst element, every
// operand strided), with the one store calling xla_max, xla_min or a compare of the flushed inputs. The function
// below is the same computation on the CPU, and the marker the packer looks for.
#include "../rd_pack.h"
#include "../xla_compare.h"

#include <cmath>
#include <cstring>

namespace {

using namespace ggml_rd;

// One marker per kind; the op's userdata points at its own.
const char kMarkers[GGML_RD_XLA_COUNT] = { 'M', 'm', '>', 'g', '<', 'l', '=', '!' };

const char *const kKernels[GGML_RD_XLA_COUNT] = { "max_xla_f32", "min_xla_f32", "cmp_gt_xla_f32", "cmp_ge_xla_f32",
	"cmp_lt_xla_f32", "cmp_le_xla_f32", "cmp_eq_xla_f32", "cmp_ne_xla_f32" };

float daz(float x) {
	uint32_t u = 0;
	std::memcpy(&u, &x, 4);
	if ((u & 0x7f800000u) == 0) {
		u &= 0x80000000u;
		std::memcpy(&x, &u, 4);
	}
	return x;
}

float one(int kind, float a0, float b0) {
	const float a = daz(a0);
	const float b = daz(b0);
	switch (kind) {
		case GGML_RD_XLA_MAX:
		case GGML_RD_XLA_MIN: {
			if (std::isnan(a)) {
				return a;
			}
			if (std::isnan(b)) {
				return b;
			}
			const bool is_min = kind == GGML_RD_XLA_MIN;
			if (a == b) {
				return std::signbit(a) ? (is_min ? a : b) : (is_min ? b : a);
			}
			return is_min ? (a < b ? a : b) : (a > b ? a : b);
		}
		case GGML_RD_XLA_GT:
			return a > b ? 1.0f : 0.0f;
		case GGML_RD_XLA_GE:
			return a >= b ? 1.0f : 0.0f;
		case GGML_RD_XLA_LT:
			return a < b ? 1.0f : 0.0f;
		case GGML_RD_XLA_LE:
			return a <= b ? 1.0f : 0.0f;
		case GGML_RD_XLA_EQ:
			return a == b ? 1.0f : 0.0f;
		default:
			return a != b ? 1.0f : 0.0f;
	}
}

struct CustomParams { // ggml_map_custom2_op_params in ggml.c
	void *fun;
	int n_tasks;
	void *userdata;
};

int kind_of(const ggml_tensor *op) {
	CustomParams p;
	std::memcpy(&p, op->op_params, sizeof(p));
	if (p.fun != reinterpret_cast<void *>(&ggml_rd_xla_binary)) {
		return -1;
	}
	const char *m = static_cast<const char *>(p.userdata);
	return m >= kMarkers && m < kMarkers + GGML_RD_XLA_COUNT ? int(m - kMarkers) : -1;
}

bool supports_xla_binary(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || b == nullptr || op->src[2] != nullptr || kind_of(op) < 0) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(a, op) || !ggml_can_repeat(b, a)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(b);
}

bool pack_xla_binary(Pack &p) {
	const int k = kind_of(p.node);
	p.kernel = k < 0 ? -1 : kernel_index(kKernels[k]);
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(xla_binary_f32, GGML_OP_MAP_CUSTOM2, -1, supports_xla_binary, pack_xla_binary);

extern "C" {

float ggml_rd_xla_binary_f32(enum ggml_rd_xla_binary_kind kind, float a, float b) { return one(int(kind), a, b); }

void ggml_rd_xla_binary(struct ggml_tensor *dst, const struct ggml_tensor *a, const struct ggml_tensor *b, int ith, int nth, void *userdata) {
	(void)nth;
	if (ith != 0) {
		return;
	}
	const char *m = static_cast<const char *>(userdata);
	const int kind = int(m - kMarkers);
	for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
		for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
			for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
				for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
					const char *pa = static_cast<const char *>(a->data) + i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3];
					const char *pb = static_cast<const char *>(b->data) + (i0 % b->ne[0]) * b->nb[0] + (i1 % b->ne[1]) * b->nb[1]
							+ (i2 % b->ne[2]) * b->nb[2] + (i3 % b->ne[3]) * b->nb[3];
					char *pd = static_cast<char *>(dst->data) + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
					float x = 0.0f, y = 0.0f;
					std::memcpy(&x, pa, 4);
					std::memcpy(&y, pb, 4);
					const float r = one(kind, x, y);
					std::memcpy(pd, &r, 4);
				}
			}
		}
	}
}

struct ggml_tensor *ggml_rd_xla_binary_op(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, enum ggml_rd_xla_binary_kind kind) {
	return ggml_map_custom2(ctx, a, b, ggml_rd_xla_binary, 1, const_cast<char *>(&kMarkers[int(kind)]));
}

} // extern "C"
