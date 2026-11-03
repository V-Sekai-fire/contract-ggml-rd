#include "stablehlo.h"
#include "../ggml-rd/rsqrt_zen2.h"
#include "../ggml-rd/xla_compare.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "ggml.h"

namespace pjrt_rd {

namespace {

// Element types. i32 and i1 values exist only at compile time (iota, constants, compares of them, masks):
// they are folded on the host and never become a ggml tensor. GGML_TYPE_I8 stands for i1.
const ggml_type kPred = GGML_TYPE_I8;

bool is_float(ggml_type t) {
	return t == GGML_TYPE_F32 || t == GGML_TYPE_BF16;
}

struct Val {
	ggml_tensor *t = nullptr; // null for an i32 or i1 value, which is always known
	std::vector<int64_t> shape; // row-major, as StableHLO writes it
	ggml_type type = GGML_TYPE_F32;
	bool neg_inf = false; // a constant -inf, or a reshape or broadcast of one
	ggml_tensor *pre = nullptr; // for a broadcast: the operand before it, shaped like the broadcast with 1s
	std::vector<int64_t> pre_shape;
	bool known = false; // the value is known at compile time: hv holds it, row-major
	std::vector<double> hv; // exact for f32, bf16, i32 and i1
};

struct Ty {
	std::vector<int64_t> shape;
	ggml_type type = GGML_TYPE_F32;
};

bool starts_with_unsupported(const std::string &e) {
	return e.compare(0, 11, "unsupported") == 0;
}

struct Lowering {
	ggml_context *ctx = nullptr;
	std::map<std::string, Val> vals;
	std::vector<ggml_tensor *> params;
	std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> consts;
	std::string err;

	bool fail(const std::string &e) {
		if (err.empty()) {
			err = e;
		}
		return false;
	}
	bool unsupported(const std::string &e) {
		return fail("unsupported: " + e);
	}
	// A type error keeps its "unsupported" marker, which picks the error code, outside the context.
	bool type_error(const std::string &context, const std::string &e) {
		return starts_with_unsupported(e) ? fail(e) : fail(context + ": " + e);
	}
};

std::string trim(const std::string &s) {
	size_t a = 0;
	size_t b = s.size();
	while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
		++a;
	}
	while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
		--b;
	}
	return s.substr(a, b - a);
}

bool starts_with(const std::string &s, const char *p) {
	return s.compare(0, std::strlen(p), p) == 0;
}

bool parse_type_at(const std::string &s, size_t at, Ty &out, size_t &end, std::string &err) {
	const size_t lt = at + 7; // after "tensor<"
	const size_t gt = s.find('>', lt);
	if (s.compare(at, 7, "tensor<") != 0 || gt == std::string::npos) {
		err = "expected a tensor type at '" + s.substr(at, 24) + "'";
		return false;
	}
	std::string body = s.substr(lt, gt - lt);
	end = gt + 1;
	out.shape.clear();
	std::vector<std::string> parts;
	size_t p = 0;
	while (true) {
		const size_t x = body.find('x', p);
		if (x == std::string::npos) {
			parts.push_back(body.substr(p));
			break;
		}
		parts.push_back(body.substr(p, x - p));
		p = x + 1;
	}
	const std::string elem = parts.back();
	if (elem == "f32") {
		out.type = GGML_TYPE_F32;
	} else if (elem == "bf16") {
		out.type = GGML_TYPE_BF16;
	} else if (elem == "i32") {
		out.type = GGML_TYPE_I32;
	} else if (elem == "i1") {
		out.type = kPred;
	} else {
		err = "unsupported: element type '" + elem + "' (f32, bf16, and i32 and i1 known at compile time)";
		return false;
	}
	for (size_t i = 0; i + 1 < parts.size(); ++i) {
		char *e = nullptr;
		const long long d = std::strtoll(parts[i].c_str(), &e, 10);
		if (parts[i].empty() || *e != '\0' || d <= 0) {
			err = "unsupported: dimension '" + parts[i] + "' (static shapes only)";
			return false;
		}
		out.shape.push_back(d);
	}
	// A ggml tensor has four dimensions. A higher rank is kept when its dimensions of size one can be dropped
	// (ne_of), which is what attention's rank-5 intermediates need.
	size_t wide = 0;
	for (int64_t d : out.shape) {
		wide += d > 1 ? 1 : 0;
	}
	if (wide > GGML_MAX_DIMS) {
		err = "unsupported: more than 4 dimensions of size above one";
		return false;
	}
	return true;
}

int64_t count_of(const std::vector<int64_t> &shape) {
	int64_t n = 1;
	for (int64_t d : shape) {
		n *= d;
	}
	return n;
}

// ggml ne of a row-major shape: the dimensions reversed. A shape of rank above 4 drops its dimensions of size
// one (parse_type_at has checked that at most four are left); every tensor of such a value is stored that way.
void ne_of(const std::vector<int64_t> &shape, int64_t ne[GGML_MAX_DIMS]) {
	for (int i = 0; i < GGML_MAX_DIMS; ++i) {
		ne[i] = 1;
	}
	std::vector<int64_t> s;
	for (int64_t d : shape) {
		if (shape.size() <= GGML_MAX_DIMS || d != 1) {
			s.push_back(d);
		}
	}
	const size_t r = s.size();
	for (size_t i = 0; i < r && i < GGML_MAX_DIMS; ++i) {
		ne[i] = s[r - 1 - i];
	}
}

// The shape without the dimensions that keep[i] does not name.
std::vector<int64_t> kept_dims(const std::vector<int64_t> &shape, const std::vector<bool> &keep) {
	std::vector<int64_t> o;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (keep[i]) {
			o.push_back(shape[i]);
		}
	}
	return o;
}

ggml_tensor *new_tensor(Lowering &L, ggml_type type, const std::vector<int64_t> &shape) {
	int64_t ne[GGML_MAX_DIMS];
	ne_of(shape, ne);
	return ggml_new_tensor_4d(L.ctx, type, ne[0], ne[1], ne[2], ne[3]);
}

// Host arrays, row-major. Calls f(index, flat) for every element of shape, in order.
template <typename F>
void each_index(const std::vector<int64_t> &shape, F f) {
	const int64_t n = count_of(shape);
	std::vector<int64_t> idx(shape.size(), 0);
	for (int64_t flat = 0; flat < n; ++flat) {
		f(idx, size_t(flat));
		for (size_t k = shape.size(); k-- > 0;) {
			if (++idx[k] < shape[k]) {
				break;
			}
			idx[k] = 0;
		}
	}
}

std::vector<int64_t> strides_of(const std::vector<int64_t> &shape) {
	std::vector<int64_t> s(shape.size(), 1);
	for (size_t k = shape.size(); k-- > 1;) {
		s[k - 1] = s[k] * shape[k];
	}
	return s;
}

// out dimension i is input dimension perm[i].
std::vector<double> h_transpose(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &perm) {
	std::vector<int64_t> os;
	for (int64_t p : perm) {
		os.push_back(shape[size_t(p)]);
	}
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(in.size());
	each_index(os, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t i = 0; i < o.size(); ++i) {
			at += o[i] * is[size_t(perm[i])];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

std::vector<double> h_broadcast(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &dims, const std::vector<int64_t> &to) {
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(size_t(count_of(to)));
	each_index(to, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t k = 0; k < dims.size(); ++k) {
			at += shape[k] == 1 ? 0 : o[size_t(dims[k])] * is[k];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

std::vector<double> h_slice(const std::vector<double> &in, const std::vector<int64_t> &shape, const std::vector<int64_t> &start,
		const std::vector<int64_t> &stride, const std::vector<int64_t> &count) {
	const std::vector<int64_t> is = strides_of(shape);
	std::vector<double> out(size_t(count_of(count)));
	each_index(count, [&](const std::vector<int64_t> &o, size_t flat) {
		int64_t at = 0;
		for (size_t i = 0; i < o.size(); ++i) {
			at += (start[i] + o[i] * stride[i]) * is[i];
		}
		out[flat] = in[size_t(at)];
	});
	return out;
}

// Rounds a value to what the element type can hold, the way a store of it would.
double h_round(ggml_type type, double v) {
	if (type == GGML_TYPE_F32) {
		return double(float(v));
	}
	if (type == GGML_TYPE_BF16) {
		return double(ggml_bf16_to_fp32(ggml_fp32_to_bf16(float(v))));
	}
	return v;
}

// An i32 operation's result, wrapped to 32 bits as two's complement.
double h_wrap32(int64_t v) {
	return double(int32_t(uint32_t(uint64_t(v))));
}

ggml_tensor *contiguous(Lowering &L, ggml_tensor *t) {
	return ggml_is_contiguous(t) ? t : ggml_cont(L.ctx, t);
}

Val reshaped(Lowering &L, const Val &v, const std::vector<int64_t> &shape) {
	Val o;
	if (v.t != nullptr) {
		int64_t ne[GGML_MAX_DIMS];
		ne_of(shape, ne);
		o.t = ggml_reshape_4d(L.ctx, contiguous(L, v.t), ne[0], ne[1], ne[2], ne[3]);
	}
	o.shape = shape;
	o.type = v.type;
	o.neg_inf = v.neg_inf;
	o.known = v.known;
	o.hv = v.hv;
	return o;
}

// A value known at compile time. A float one also gets a constant tensor; an i32 or i1 one stays on the host.
Val known_val(Lowering &L, ggml_type type, const std::vector<int64_t> &shape, std::vector<double> hv) {
	Val v;
	v.shape = shape;
	v.type = type;
	v.known = true;
	v.hv = std::move(hv);
	if (is_float(type)) {
		v.t = new_tensor(L, type, shape);
		const size_t esz = type == GGML_TYPE_F32 ? 4 : 2;
		std::vector<uint8_t> bytes(v.hv.size() * esz);
		bool all_neg_inf = !v.hv.empty();
		for (size_t i = 0; i < v.hv.size(); ++i) {
			const float f = float(v.hv[i]);
			all_neg_inf = all_neg_inf && std::isinf(f) && f < 0.0f;
			if (type == GGML_TYPE_F32) {
				std::memcpy(bytes.data() + i * 4, &f, 4);
			} else {
				const ggml_bf16_t b = ggml_fp32_to_bf16(f);
				std::memcpy(bytes.data() + i * 2, &b, 2);
			}
		}
		L.consts[v.t] = std::move(bytes);
		v.neg_inf = all_neg_inf;
	}
	return v;
}

Val as_type(Lowering &L, const Val &v, ggml_type type) {
	if (v.type == type) {
		return v;
	}
	if ((v.type == kPred && type == GGML_TYPE_F32) || (v.type == GGML_TYPE_F32 && type == kPred)) {
		// An i1 value computed at run time is an f32 tensor of 0 and 1: only the label changes.
		Val o = v;
		o.type = type;
		return o;
	}
	Val o = v;
	o.t = ggml_cast(L.ctx, v.t, type);
	o.type = type;
	if (o.known) {
		for (double &x : o.hv) {
			x = h_round(type, x);
		}
	}
	return o;
}

// out dimension i is input dimension perm[i], as stablehlo.transpose says.
Val transposed(Lowering &L, const Val &v, const std::vector<int64_t> &perm) {
	const int r = int(v.shape.size());
	bool identity = true;
	for (int i = 0; i < r; ++i) {
		identity = identity && perm[size_t(i)] == i;
	}
	if (identity) {
		return v;
	}
	std::vector<int64_t> out_shape;
	for (int i = 0; i < r; ++i) {
		out_shape.push_back(v.shape[size_t(perm[size_t(i)])]);
	}
	if (v.t == nullptr || r > GGML_MAX_DIMS) {
		Val o;
		if (v.t != nullptr) {
			// Above rank 4 the tensor holds only the dimensions of size above one (ne_of); permute those.
			std::vector<int64_t> at(size_t(r), -1);
			Val s = v;
			s.shape.clear();
			s.known = false;
			for (int i = 0; i < r; ++i) {
				if (v.shape[size_t(i)] != 1) {
					at[size_t(i)] = int64_t(s.shape.size());
					s.shape.push_back(v.shape[size_t(i)]);
				}
			}
			std::vector<int64_t> sp;
			for (int i = 0; i < r; ++i) {
				if (at[size_t(perm[size_t(i)])] >= 0) {
					sp.push_back(at[size_t(perm[size_t(i)])]);
				}
			}
			o = transposed(L, s, sp);
		}
		o.shape = out_shape;
		o.type = v.type;
		o.neg_inf = v.neg_inf;
		if (v.known) {
			o.known = true;
			o.hv = h_transpose(v.hv, v.shape, perm);
		}
		return o;
	}
	int axis[GGML_MAX_DIMS] = { 0, 1, 2, 3 };
	for (int k = 0; k < r; ++k) {
		const int hlo_in = r - 1 - k;
		for (int i = 0; i < r; ++i) {
			if (perm[size_t(i)] == hlo_in) {
				axis[k] = r - 1 - i;
			}
		}
	}
	Val o;
	o.t = ggml_cont(L.ctx, ggml_permute(L.ctx, v.t, axis[0], axis[1], axis[2], axis[3]));
	o.type = v.type;
	o.shape = out_shape;
	o.neg_inf = v.neg_inf;
	if (v.known) {
		o.known = true;
		o.hv = h_transpose(v.hv, v.shape, perm);
	}
	return o;
}

bool int_list(const std::string &s, const char *key, size_t from, std::vector<int64_t> &out, size_t *after) {
	const size_t k = s.find(key, from);
	if (k == std::string::npos) {
		return false;
	}
	const size_t lb = s.find('[', k);
	const size_t rb = s.find(']', lb);
	if (lb == std::string::npos || rb == std::string::npos) {
		return false;
	}
	out.clear();
	size_t p = lb + 1;
	while (p < rb) {
		while (p < rb && (s[p] == ' ' || s[p] == ',')) {
			++p;
		}
		if (p >= rb) {
			break;
		}
		char *e = nullptr;
		out.push_back(std::strtoll(s.c_str() + p, &e, 10));
		p = size_t(e - s.c_str());
	}
	if (after != nullptr) {
		*after = rb + 1;
	}
	return true;
}

// "name = [a, b] x [c, d]"
bool int_list_pair(const std::string &s, const char *key, std::vector<int64_t> &lhs, std::vector<int64_t> &rhs) {
	size_t after = 0;
	if (!int_list(s, key, 0, lhs, &after)) {
		return false;
	}
	const size_t x = s.find('x', after);
	const size_t lb = s.find('[', after);
	if (x == std::string::npos || lb == std::string::npos || x > lb) {
		return false;
	}
	return int_list(s, "[", after, rhs, nullptr);
}

// An f32 constant of the given shape with every element v.
Val filled(Lowering &L, const std::vector<int64_t> &shape, float v) {
	Val o;
	o.t = new_tensor(L, GGML_TYPE_F32, shape);
	o.shape = shape;
	o.type = GGML_TYPE_F32;
	std::vector<uint8_t> bytes(size_t(count_of(shape)) * 4);
	for (size_t i = 0; i < bytes.size() / 4; ++i) {
		std::memcpy(bytes.data() + i * 4, &v, 4);
	}
	L.consts[o.t] = std::move(bytes);
	return o;
}

// An f32 constant tensor with like's ne, every element v.
ggml_tensor *filled_like(Lowering &L, const ggml_tensor *like, float v) {
	ggml_tensor *t = ggml_new_tensor_4d(L.ctx, GGML_TYPE_F32, like->ne[0], like->ne[1], like->ne[2], like->ne[3]);
	std::vector<uint8_t> bytes(size_t(ggml_nelements(t)) * 4);
	for (size_t i = 0; i < bytes.size() / 4; ++i) {
		std::memcpy(bytes.data() + i * 4, &v, 4);
	}
	L.consts[t] = std::move(bytes);
	return t;
}

bool op_unary(Lowering &L, const std::string &op, const Val &a, Val &out) {
	Val f = as_type(L, a, GGML_TYPE_F32);
	Val o = f;
	if (op == "negate") {
		o.t = ggml_neg(L.ctx, f.t);
	} else if (op == "sine") {
		o.t = ggml_sin(L.ctx, f.t);
	} else if (op == "cosine") {
		o.t = ggml_cos(L.ctx, f.t);
	} else if (op == "exponential") {
		o.t = ggml_exp(L.ctx, f.t);
	} else if (op == "log") {
		o.t = ggml_log(L.ctx, f.t);
	} else if (op == "sqrt") {
		o.t = ggml_sqrt(L.ctx, f.t);
	} else if (op == "abs") {
		o.t = ggml_abs(L.ctx, f.t);
	} else if (op == "floor") {
		o.t = ggml_floor(L.ctx, f.t);
	} else if (op == "sign") {
		// SGN gives a zero for a zero, but not a signed one: ggml-cpu returns +0 and ggml-rd -0, and XLA returns the
		// zero it was given. Rebuild the sign with exact arithmetic that does not depend on it:
		//   s = sgn(x), ss = s*s (1 for a nonzero, +0 for a zero), c = s + sgn(1/x) * (1 - ss), result = c * ss.
		// c is +-1 with the sign of x everywhere (1/+-0 is +-inf), so c * (+0) at a zero is a zero with x's sign.
		// A NaN gives NaN (see below).
		const Val ones = filled(L, f.shape, 1.0f);
		ggml_tensor *sg = ggml_sgn(L.ctx, f.t);
		ggml_tensor *ss = ggml_mul(L.ctx, sg, sg);
		ggml_tensor *q = ggml_sgn(L.ctx, ggml_div(L.ctx, ones.t, f.t));
		ggml_tensor *c = ggml_add(L.ctx, sg, ggml_mul(L.ctx, q, ggml_sub(L.ctx, ones.t, ss)));
		// A NaN is the one input where ss = 0 and sgn(1/x) = 0 (a zero has sgn(1/x) = +-1, an infinity has ss = 1).
		// ind = (1 - ss) * (1 - e*e) is 1 only there, and 1 + 0/(1 - ind) is NaN there (0/0) and 1 elsewhere.
		const Val zeros = filled(L, f.shape, 0.0f);
		ggml_tensor *e2 = ggml_mul(L.ctx, q, q);
		ggml_tensor *ind = ggml_mul(L.ctx, ggml_sub(L.ctx, ones.t, ss), ggml_sub(L.ctx, ones.t, e2));
		ggml_tensor *nan_or_one = ggml_add(L.ctx, ones.t, ggml_div(L.ctx, zeros.t, ggml_sub(L.ctx, ones.t, ind)));
		o.t = ggml_mul(L.ctx, ggml_mul(L.ctx, c, ss), nan_or_one);
	} else if (op == "rsqrt") {
		// XLA CPU's rsqrt is the CPU's estimate instruction and one Newton step; this is that on a Zen 2 (rsqrt_zen2.h).
		o.t = ggml_rd_rsqrt_zen2_op(L.ctx, contiguous(L, f.t));
	} else {
		return L.unsupported("stablehlo." + op);
	}
	out = as_type(L, o, a.type);
	return true;
}

bool op_binary(Lowering &L, const std::string &op, const Val &a, const Val &b, Val &out) {
	if (a.shape != b.shape) {
		return L.fail("stablehlo." + op + ": operand shapes differ");
	}
	if (op == "maximum") {
		// Softmax writes max(-inf, row max); a maximum against a constant -inf returns the other operand, NaN included.
		if (a.neg_inf || b.neg_inf) {
			out = a.neg_inf ? b : a;
			return true;
		}
		return L.unsupported("stablehlo.maximum of two values (only against a constant -inf)");
	}
	Val fa = as_type(L, a, GGML_TYPE_F32);
	Val fb = as_type(L, b, GGML_TYPE_F32);
	Val o = fa;
	if (op == "add") {
		o.t = ggml_add(L.ctx, fa.t, fb.t);
	} else if (op == "subtract") {
		o.t = ggml_sub(L.ctx, fa.t, fb.t);
	} else if (op == "multiply") {
		o.t = ggml_mul(L.ctx, fa.t, fb.t);
	} else if (op == "divide") {
		if (fb.pre != nullptr && fb.type == GGML_TYPE_F32 && fb.pre->type == GGML_TYPE_F32) {
			// XLA's algebraic simplifier rewrites a / broadcast(b) as a * broadcast(1 / b), which rounds twice.
			// Doing the same keeps the bits: for a divisor of 3 the two results differ in the last place.
			ggml_tensor *rcp = ggml_div(L.ctx, filled_like(L, fb.pre, 1.0f), fb.pre);
			o.t = ggml_mul(L.ctx, fa.t, ggml_repeat(L.ctx, rcp, fa.t));
		} else {
			o.t = ggml_div(L.ctx, fa.t, fb.t);
		}
	} else {
		return L.unsupported("stablehlo." + op);
	}
	out = as_type(L, o, a.type);
	return true;
}

float element_f32(const Val &c, const std::vector<uint8_t> &bytes) {
	if (c.type == GGML_TYPE_F32) {
		float f = 0.0f;
		std::memcpy(&f, bytes.data(), 4);
		return f;
	}
	ggml_bf16_t b;
	std::memcpy(&b, bytes.data(), 2);
	return ggml_bf16_to_fp32(b);
}

// stablehlo.reduce of add or maximum over some dimensions, with the operation's identity as init.
bool op_reduce(Lowering &L, const std::string &attrs, const Val &x, const Val &init, const Ty &to, Val &out) {
	const size_t ap = attrs.find("applies stablehlo.");
	if (ap == std::string::npos) {
		return L.unsupported("stablehlo.reduce with a body other than a named stablehlo op");
	}
	const size_t ks = ap + 18;
	const size_t ke = attrs.find(' ', ks);
	const std::string kind = attrs.substr(ks, ke == std::string::npos ? std::string::npos : ke - ks);
	if (kind != "add" && kind != "maximum") {
		return L.unsupported("stablehlo.reduce applying stablehlo." + kind);
	}
	std::vector<int64_t> dims;
	if (!int_list(attrs, "dimensions", 0, dims, nullptr)) {
		return L.fail("stablehlo.reduce: no dimensions");
	}
	const int64_t r = int64_t(x.shape.size());
	std::vector<bool> reduced(size_t(r), false);
	for (int64_t d : dims) {
		if (d < 0 || d >= r || reduced[size_t(d)]) {
			return L.fail("stablehlo.reduce: dimension " + std::to_string(d) + " is out of range or repeated");
		}
		reduced[size_t(d)] = true;
	}
	std::unordered_map<const ggml_tensor *, std::vector<uint8_t> >::const_iterator ci = L.consts.find(init.t);
	if (ci == L.consts.end() || !init.shape.empty()) {
		return L.unsupported("stablehlo.reduce with an init that is not a scalar constant");
	}
	const float iv = element_f32(init, ci->second);
	const bool identity = kind == "add" ? (iv == 0.0f && !std::signbit(iv)) : (std::isinf(iv) && iv < 0.0f);
	if (!identity) {
		return L.unsupported("stablehlo.reduce whose init is not the identity of " + kind);
	}
	std::vector<int64_t> kept, perm, flat;
	int64_t n = 1;
	for (int64_t i = 0; i < r; ++i) {
		if (!reduced[size_t(i)]) {
			kept.push_back(x.shape[size_t(i)]);
			perm.push_back(i);
		}
	}
	for (int64_t i = 0; i < r; ++i) {
		if (reduced[size_t(i)]) {
			perm.push_back(i);
			n *= x.shape[size_t(i)];
		}
	}
	if (kept != to.shape) {
		return L.fail("stablehlo.reduce: the result shape does not match the kept dimensions");
	}
	if (x.t == nullptr) {
		return L.unsupported("stablehlo.reduce of i32 or i1 values");
	}
	if (n == 1 && kept.size() + 1 > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.reduce over dimensions of size one above rank 4");
	}
	Val v = as_type(L, x, GGML_TYPE_F32);
	v = transposed(L, v, perm);
	flat = kept;
	flat.push_back(n);
	v = reshaped(L, v, flat);
	Val o;
	o.type = GGML_TYPE_F32;
	if (kind == "add") {
		o.t = ggml_sum_rows(L.ctx, v.t);
	} else {
		o.t = ggml_pool_1d(L.ctx, v.t, GGML_OP_POOL_MAX, int(n), int(n), 0);
	}
	std::vector<int64_t> rows = kept;
	rows.push_back(1);
	o.shape = rows;
	o = reshaped(L, o, kept);
	out = as_type(L, o, x.type);
	return true;
}

// stablehlo.slice %x [start:limit(:stride), ...]: a strided view, copied out.
bool slice_core(Lowering &L, const Val &x, const std::vector<int64_t> &start, const std::vector<int64_t> &stride,
		const std::vector<int64_t> &span, const std::vector<int64_t> &count, Val &out);

bool op_slice(Lowering &L, const std::string &attrs, const Val &x, const Ty &to, Val &out) {
	const size_t lb = attrs.find('[');
	const size_t rb = attrs.find(']', lb);
	if (lb == std::string::npos || rb == std::string::npos) {
		return L.fail("stablehlo.slice: no index ranges");
	}
	std::vector<int64_t> start, count, stride, span;
	size_t p = lb + 1;
	while (p < rb) {
		while (p < rb && (attrs[p] == ' ' || attrs[p] == ',')) {
			++p;
		}
		if (p >= rb) {
			break;
		}
		char *e = nullptr;
		const int64_t a = std::strtoll(attrs.c_str() + p, &e, 10);
		if (*e != ':') {
			return L.fail("stablehlo.slice: cannot read a start:limit range");
		}
		const int64_t b = std::strtoll(e + 1, &e, 10);
		int64_t st = 1;
		if (*e == ':') {
			st = std::strtoll(e + 1, &e, 10);
		}
		if (st < 1 || a < 0 || b < a) {
			return L.fail("stablehlo.slice: a range is empty, negative or has a stride below one");
		}
		start.push_back(a);
		stride.push_back(st);
		span.push_back(b - a);
		count.push_back((b - a + st - 1) / st);
		p = size_t(e - attrs.c_str());
	}
	const size_t r = x.shape.size();
	if (start.size() != r) {
		return L.fail("stablehlo.slice: " + std::to_string(start.size()) + " ranges for rank " + std::to_string(r));
	}
	for (size_t i = 0; i < r; ++i) {
		if (start[i] + (count[i] - 1) * stride[i] >= x.shape[i] && count[i] > 0) {
			return L.fail("stablehlo.slice: a range is past the end of its dimension");
		}
	}
	(void)to;
	if (x.t == nullptr) {
		out = Val();
		out.shape = count;
		out.type = x.type;
		out.known = true;
		out.hv = h_slice(x.hv, x.shape, start, stride, count);
		return true;
	}
	Val c = x;
	if (r > GGML_MAX_DIMS) {
		// Above rank 4 a dimension of size one can only be sliced [0:1]; slice the others.
		std::vector<bool> keep(r);
		for (size_t i = 0; i < r; ++i) {
			keep[i] = x.shape[i] != 1;
		}
		c.shape = kept_dims(x.shape, keep);
		c.known = false;
		Val o;
		if (!slice_core(L, c, kept_dims(start, keep), kept_dims(stride, keep), kept_dims(span, keep), kept_dims(count, keep), o)) {
			return false;
		}
		o = reshaped(L, o, count);
		if (x.known) {
			o.known = true;
			o.hv = h_slice(x.hv, x.shape, start, stride, count);
		}
		out = o;
		return true;
	}
	if (!slice_core(L, c, start, stride, span, count, out)) {
		return false;
	}
	if (x.known) {
		out.known = true;
		out.hv = h_slice(x.hv, x.shape, start, stride, count);
	}
	return true;
}

bool slice_core(Lowering &L, const Val &x, const std::vector<int64_t> &start, const std::vector<int64_t> &stride,
		const std::vector<int64_t> &span, const std::vector<int64_t> &count, Val &out) {
	const size_t r = x.shape.size();
	Val c = x;
	c.t = contiguous(L, x.t);
	// A ggml view cannot step along its first dimension (nb0 is the element size), so a stride on the
	// last StableHLO dimension is applied afterwards, on a transposed copy.
	const bool last_strided = r > 0 && stride[r - 1] > 1;
	int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
	size_t nb[GGML_MAX_DIMS] = { c.t->nb[0], c.t->nb[1], c.t->nb[2], c.t->nb[3] };
	size_t offset = 0;
	for (size_t i = 0; i < r; ++i) {
		const size_t k = r - 1 - i; // ggml dimension of StableHLO dimension i
		const bool defer = last_strided && i == r - 1;
		ne[k] = defer ? span[i] : count[i];
		nb[k] = c.t->nb[k] * (defer ? size_t(1) : size_t(stride[i]));
		offset += size_t(start[i]) * c.t->nb[k];
	}
	Val o;
	o.t = ggml_cont(L.ctx, ggml_view_4d(L.ctx, c.t, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset));
	if (last_strided) {
		ggml_tensor *t = ggml_cont(L.ctx, ggml_permute(L.ctx, o.t, 1, 0, 2, 3));
		ggml_tensor *v = ggml_view_4d(L.ctx, t, t->ne[0], count[r - 1], t->ne[2], t->ne[3], t->nb[1] * size_t(stride[r - 1]), t->nb[2], t->nb[3], 0);
		o.t = ggml_cont(L.ctx, ggml_permute(L.ctx, ggml_cont(L.ctx, v), 1, 0, 2, 3));
	}
	o.shape = count;
	o.type = x.type;
	out = o;
	return true;
}

bool op_concatenate(Lowering &L, const std::string &attrs, const std::vector<Val> &in, Val &out) {
	const size_t d = attrs.find("dim = ");
	if (d == std::string::npos || in.empty()) {
		return L.fail("stablehlo.concatenate: no dim");
	}
	const int64_t dim = std::strtoll(attrs.c_str() + d + 6, nullptr, 10);
	const int64_t r = int64_t(in[0].shape.size());
	if (dim < 0 || dim >= r) {
		return L.fail("stablehlo.concatenate: dim is out of range");
	}
	std::vector<int64_t> out_shape = in[0].shape;
	bool all_known = in[0].known;
	for (size_t k = 1; k < in.size(); ++k) {
		if (in[k].type != in[0].type || in[k].shape.size() != in[0].shape.size()) {
			return L.fail("stablehlo.concatenate: operands differ in rank or element type");
		}
		for (int64_t i = 0; i < r; ++i) {
			if (i != dim && in[k].shape[size_t(i)] != in[0].shape[size_t(i)]) {
				return L.fail("stablehlo.concatenate: operands differ outside the concatenated dimension");
			}
		}
		out_shape[size_t(dim)] += in[k].shape[size_t(dim)];
		all_known = all_known && in[k].known;
	}
	std::vector<double> hv;
	if (all_known) {
		// Row-major: the outer dimensions in turn, each operand's block of the inner ones.
		int64_t outer = 1;
		for (int64_t i = 0; i < dim; ++i) {
			outer *= out_shape[size_t(i)];
		}
		for (int64_t o = 0; o < outer; ++o) {
			for (const Val &v : in) {
				const size_t block = size_t(count_of(v.shape) / outer);
				hv.insert(hv.end(), v.hv.begin() + std::ptrdiff_t(size_t(o) * block), v.hv.begin() + std::ptrdiff_t(size_t(o + 1) * block));
			}
		}
	}
	if (in[0].t == nullptr) {
		out = Val();
		out.shape = out_shape;
		out.type = in[0].type;
		out.known = true;
		out.hv = std::move(hv);
		return true;
	}
	// Above rank 4, concatenate on the dimensions of size above one and the concatenated one.
	std::vector<bool> keep(size_t(r), true);
	int64_t gdim = dim;
	if (r > GGML_MAX_DIMS) {
		gdim = 0;
		for (int64_t i = 0; i < r; ++i) {
			keep[size_t(i)] = i == dim || out_shape[size_t(i)] != 1;
			gdim += i < dim && keep[size_t(i)] ? 1 : 0;
		}
		if (kept_dims(out_shape, keep).size() > GGML_MAX_DIMS) {
			return L.unsupported("stablehlo.concatenate of more than 4 dimensions of size above one");
		}
	}
	const int64_t gr = int64_t(kept_dims(out_shape, keep).size());
	const bool squeeze = r > GGML_MAX_DIMS;
	Val acc = squeeze ? reshaped(L, in[0], kept_dims(in[0].shape, keep)) : in[0];
	for (size_t k = 1; k < in.size(); ++k) {
		const Val next = squeeze ? reshaped(L, in[k], kept_dims(in[k].shape, keep)) : in[k];
		Val f0 = as_type(L, acc, GGML_TYPE_F32);
		Val f1 = as_type(L, next, GGML_TYPE_F32);
		Val o = acc;
		o.t = as_type(L, Val{ ggml_concat(L.ctx, contiguous(L, f0.t), contiguous(L, f1.t), int(gr - 1 - gdim)), f0.shape, GGML_TYPE_F32 }, acc.type).t;
		o.shape[size_t(gdim)] += next.shape[size_t(gdim)];
		o.known = false;
		acc = o;
	}
	out = squeeze ? reshaped(L, acc, out_shape) : acc;
	out.neg_inf = false;
	if (all_known) {
		out.known = true;
		out.hv = std::move(hv);
	}
	return true;
}

// stablehlo.pad with a zero padding value, non-negative edges and no interior padding.
bool op_pad(Lowering &L, const std::string &attrs, const Val &x, const Val &pad_value, Val &out) {
	std::vector<int64_t> low, high, interior;
	if (!int_list(attrs, "low", 0, low, nullptr) || !int_list(attrs, "high", 0, high, nullptr) || !int_list(attrs, "interior", 0, interior, nullptr)) {
		return L.fail("stablehlo.pad: low, high and interior are all needed");
	}
	const size_t r = x.shape.size();
	if (low.size() != r || high.size() != r || interior.size() != r) {
		return L.fail("stablehlo.pad: the padding lists do not match the rank");
	}
	if (x.t == nullptr || r > GGML_MAX_DIMS) {
		return L.unsupported("stablehlo.pad of i32 or i1 values, or above rank 4");
	}
	for (size_t i = 0; i < r; ++i) {
		if (interior[i] != 0) {
			return L.unsupported("stablehlo.pad with interior padding");
		}
		if (low[i] < 0 || high[i] < 0) {
			return L.unsupported("stablehlo.pad with negative edge padding");
		}
	}
	std::unordered_map<const ggml_tensor *, std::vector<uint8_t> >::const_iterator ci = L.consts.find(pad_value.t);
	if (ci == L.consts.end() || !pad_value.shape.empty() || element_f32(pad_value, ci->second) != 0.0f) {
		return L.unsupported("stablehlo.pad with a padding value other than a constant zero");
	}
	int lp[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
	int rp[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
	for (size_t i = 0; i < r; ++i) {
		lp[r - 1 - i] = int(low[i]);
		rp[r - 1 - i] = int(high[i]);
	}
	Val f = as_type(L, x, GGML_TYPE_F32);
	Val o = f;
	o.t = ggml_pad_ext(L.ctx, contiguous(L, f.t), lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]);
	for (size_t i = 0; i < r; ++i) {
		o.shape[i] += low[i] + high[i];
	}
	out = as_type(L, o, x.type);
	return true;
}

bool op_broadcast(Lowering &L, const Val &x, const std::vector<int64_t> &dims, const Ty &to, Val &out) {
	if (dims.size() != x.shape.size()) {
		return L.fail("stablehlo.broadcast_in_dim: dims has " + std::to_string(dims.size()) + " entries for rank " + std::to_string(x.shape.size()));
	}
	std::vector<int64_t> mid(to.shape.size(), 1);
	int64_t last = -1;
	for (size_t i = 0; i < dims.size(); ++i) {
		if (dims[i] <= last || dims[i] >= int64_t(to.shape.size())) {
			return L.unsupported("stablehlo.broadcast_in_dim with dims that are not strictly increasing");
		}
		last = dims[i];
		if (x.shape[i] != to.shape[size_t(dims[i])] && x.shape[i] != 1) {
			return L.fail("stablehlo.broadcast_in_dim: dimension " + std::to_string(i) + " does not fit");
		}
		mid[size_t(dims[i])] = x.shape[i];
	}
	if (x.t == nullptr) {
		out = Val();
		out.shape = to.shape;
		out.type = x.type;
		out.known = true;
		out.hv = h_broadcast(x.hv, x.shape, dims, to.shape);
		return true;
	}
	Val r;
	if (to.shape.size() > GGML_MAX_DIMS) {
		// Above rank 4 the result drops its dimensions of size one (ne_of); drop the same ones from mid, so the
		// repeat lines up.
		std::vector<int64_t> m;
		for (size_t i = 0; i < mid.size(); ++i) {
			if (to.shape[i] != 1) {
				m.push_back(mid[i]);
			}
		}
		const Val sq = reshaped(L, x, m);
		r = sq;
		r.shape = mid;
	} else {
		r = reshaped(L, x, mid);
	}
	if (mid == to.shape) {
		out = r;
		if (x.known) {
			out.hv = x.hv;
		}
		return true;
	}
	Val o;
	int64_t ne[GGML_MAX_DIMS];
	ne_of(to.shape, ne);
	o.t = ggml_repeat_4d(L.ctx, r.t, ne[0], ne[1], ne[2], ne[3]);
	o.shape = to.shape;
	o.type = x.type;
	o.neg_inf = x.neg_inf;
	o.pre = r.t;
	o.pre_shape = mid;
	if (x.known) {
		o.known = true;
		o.hv = h_broadcast(x.hv, x.shape, dims, to.shape);
	}
	out = o;
	return true;
}

// Reshapes only when the ggml ne changes: a reshape to the same ne would add a node and change nothing.
Val reshaped_ne(Lowering &L, const Val &v, const std::vector<int64_t> &shape) {
	int64_t a[GGML_MAX_DIMS], b[GGML_MAX_DIMS];
	ne_of(v.shape, a);
	ne_of(shape, b);
	if (v.t != nullptr && ggml_is_contiguous(v.t) && std::equal(a, a + GGML_MAX_DIMS, b)) {
		Val o = v;
		o.shape = shape;
		return o;
	}
	return reshaped(L, v, shape);
}

// dot_general: the batch, free and contracting dimensions of each side are each flattened into one, so the
// product is one ggml_mul_mat of [B, M, K] by [B, N, K]; the result [B, M, N] is reshaped to the batch dims,
// the lhs free dims and the rhs free dims, the order StableHLO gives them.
bool op_dot(Lowering &L, const std::string &attrs, const Val &lhs, const Val &rhs, const Ty &to, Val &out) {
	std::vector<int64_t> lb, rb, lc, rc;
	int_list_pair(attrs, "batching_dims", lb, rb);
	if (!int_list_pair(attrs, "contracting_dims", lc, rc) || lc.empty() || lc.size() != rc.size() || lb.size() != rb.size()) {
		return L.unsupported("stablehlo.dot_general without contracting dimensions paired on both sides");
	}
	if (lhs.t == nullptr || rhs.t == nullptr) {
		return L.unsupported("stablehlo.dot_general of i32 or i1 values");
	}
	const int64_t lr = int64_t(lhs.shape.size());
	const int64_t rr = int64_t(rhs.shape.size());
	std::vector<int> luse(size_t(lr), 0), ruse(size_t(rr), 0);
	for (size_t i = 0; i < lc.size(); ++i) {
		if (lc[i] < 0 || lc[i] >= lr || rc[i] < 0 || rc[i] >= rr || luse[size_t(lc[i])]++ != 0 || ruse[size_t(rc[i])]++ != 0) {
			return L.fail("stablehlo.dot_general: a contracting dimension is out of range or repeated");
		}
		if (lhs.shape[size_t(lc[i])] != rhs.shape[size_t(rc[i])]) {
			return L.fail("stablehlo.dot_general: contracted dimensions differ");
		}
	}
	for (size_t i = 0; i < lb.size(); ++i) {
		if (lb[i] < 0 || lb[i] >= lr || rb[i] < 0 || rb[i] >= rr || luse[size_t(lb[i])]++ != 0 || ruse[size_t(rb[i])]++ != 0) {
			return L.fail("stablehlo.dot_general: a batch dimension is out of range or repeated");
		}
		if (lhs.shape[size_t(lb[i])] != rhs.shape[size_t(rb[i])]) {
			return L.fail("stablehlo.dot_general: batch dimensions differ");
		}
	}
	std::vector<int64_t> lfree, rfree;
	for (int64_t d = 0; d < lr; ++d) {
		if (luse[size_t(d)] == 0) {
			lfree.push_back(d);
		}
	}
	for (int64_t d = 0; d < rr; ++d) {
		if (ruse[size_t(d)] == 0) {
			rfree.push_back(d);
		}
	}
	int64_t B = 1, M = 1, N = 1, K = 1;
	std::vector<int64_t> want;
	for (int64_t d : lb) {
		B *= lhs.shape[size_t(d)];
		want.push_back(lhs.shape[size_t(d)]);
	}
	for (int64_t d : lfree) {
		M *= lhs.shape[size_t(d)];
		want.push_back(lhs.shape[size_t(d)]);
	}
	for (int64_t d : rfree) {
		N *= rhs.shape[size_t(d)];
		want.push_back(rhs.shape[size_t(d)]);
	}
	for (int64_t d : lc) {
		K *= lhs.shape[size_t(d)];
	}
	std::vector<int64_t> lperm = lb;
	lperm.insert(lperm.end(), lfree.begin(), lfree.end());
	lperm.insert(lperm.end(), lc.begin(), lc.end());
	std::vector<int64_t> rperm = rb;
	rperm.insert(rperm.end(), rfree.begin(), rfree.end());
	rperm.insert(rperm.end(), rc.begin(), rc.end());
	Val l = reshaped_ne(L, transposed(L, lhs, lperm), { B, M, K });
	Val r = reshaped_ne(L, transposed(L, rhs, rperm), { B, N, K });
	l = as_type(L, l, GGML_TYPE_F32);
	Val o;
	o.t = ggml_mul_mat(L.ctx, contiguous(L, r.t), contiguous(L, l.t));
	o.type = GGML_TYPE_F32;
	o.shape = { B, M, N };
	out = as_type(L, reshaped(L, o, want), to.type);
	return true;
}

bool op_constant(Lowering &L, const std::string &line, const Ty &ty, Val &out) {
	const size_t d = line.find("dense<");
	size_t e = std::string::npos;
	if (d != std::string::npos) {
		e = line.find('>', d);
	}
	if (d == std::string::npos || e == std::string::npos) {
		return L.fail("stablehlo.constant without dense<...>");
	}
	const std::string body = line.substr(d + 6, e - d - 6);
	const int64_t n = count_of(ty.shape);
	std::vector<double> values;
	if (body.compare(0, 3, "\"0x") == 0) {
		// A large constant is printed as its raw bytes, little-endian, in hex: every element, or one to splat.
		// An i1 element is one byte.
		const size_t close = body.find('"', 3);
		const size_t esz = ty.type == GGML_TYPE_F32 || ty.type == GGML_TYPE_I32 ? 4 : ty.type == GGML_TYPE_BF16 ? 2 : 1;
		if (close == std::string::npos || (close - 3) % 2 != 0) {
			return L.fail("stablehlo.constant: a hex string with an odd number of digits");
		}
		std::vector<uint8_t> raw((close - 3) / 2);
		for (size_t i = 0; i < raw.size(); ++i) {
			char hx[3] = { body[3 + 2 * i], body[4 + 2 * i], 0 };
			char *x = nullptr;
			raw[i] = uint8_t(std::strtoul(hx, &x, 16));
			if (*x != '\0') {
				return L.fail("stablehlo.constant: a hex string with a digit that is not hex");
			}
		}
		if (raw.size() != esz && raw.size() != size_t(n) * esz) {
			return L.fail("stablehlo.constant: " + std::to_string(raw.size()) + " hex bytes for " + std::to_string(n) + " elements");
		}
		for (size_t i = 0; i < raw.size(); i += esz) {
			if (ty.type == GGML_TYPE_F32) {
				float f = 0.0f;
				std::memcpy(&f, raw.data() + i, 4);
				values.push_back(double(f));
			} else if (ty.type == GGML_TYPE_I32) {
				int32_t v = 0;
				std::memcpy(&v, raw.data() + i, 4);
				values.push_back(double(v));
			} else if (ty.type == GGML_TYPE_BF16) {
				ggml_bf16_t b;
				std::memcpy(&b, raw.data() + i, 2);
				values.push_back(double(ggml_bf16_to_fp32(b)));
			} else {
				values.push_back(raw[i] != 0 ? 1.0 : 0.0);
			}
		}
	} else if (body.compare(0, 2, "0x") == 0) {
		// StableHLO writes an infinity as its hex bit pattern; any other hex literal is refused.
		const size_t digits = ty.type == GGML_TYPE_F32 ? 8 : 4;
		if (!is_float(ty.type) || body.size() != 2 + digits || !ty.shape.empty()) {
			return L.unsupported("stablehlo.constant with a hex literal other than a scalar infinity");
		}
		const uint32_t bits = uint32_t(std::strtoul(body.c_str() + 2, nullptr, 16));
		const uint32_t f32_bits = ty.type == GGML_TYPE_F32 ? bits : bits << 16;
		float f = 0.0f;
		std::memcpy(&f, &f32_bits, 4);
		if (!std::isinf(f)) {
			return L.unsupported("stablehlo.constant with a hex literal other than an infinity");
		}
		values.push_back(double(f));
	} else {
		if (body.find('"') != std::string::npos || body.find("0x") != std::string::npos || body.find("inf") != std::string::npos || body.find("nan") != std::string::npos) {
			return L.unsupported("stablehlo.constant with a hex, string, inf or nan literal");
		}
		size_t p = 0;
		while (p < body.size()) {
			const char c = body[p];
			if (c == '[' || c == ']' || c == ',' || c == ' ') {
				++p;
				continue;
			}
			if (ty.type == kPred && (body.compare(p, 4, "true") == 0 || body.compare(p, 5, "false") == 0)) {
				const bool t = body[p] == 't';
				values.push_back(t ? 1.0 : 0.0);
				p += t ? 4 : 5;
				continue;
			}
			char *end = nullptr;
			const double v = std::strtod(body.c_str() + p, &end);
			if (end == body.c_str() + p || ty.type == kPred) {
				return L.fail("stablehlo.constant: cannot read '" + body.substr(p, 12) + "'");
			}
			if (ty.type == GGML_TYPE_I32 && (v != std::floor(v) || v < -2147483648.0 || v > 2147483647.0)) {
				return L.fail("stablehlo.constant: '" + body.substr(p, 12) + "' is not an i32");
			}
			values.push_back(h_round(ty.type, v));
			p = size_t(end - body.c_str());
		}
	}
	if (values.size() != 1 && int64_t(values.size()) != n) {
		return L.fail("stablehlo.constant: " + std::to_string(values.size()) + " values for " + std::to_string(n) + " elements");
	}
	std::vector<double> hv(static_cast<size_t>(n));
	for (int64_t i = 0; i < n; ++i) {
		hv[size_t(i)] = values.size() == 1 ? values[0] : values[size_t(i)];
	}
	out = known_val(L, ty.type, ty.shape, std::move(hv));
	return true;
}

// Operands "%a, %b" at the start of s; returns the text after them.
std::vector<std::string> operands(const std::string &s, size_t &end) {
	std::vector<std::string> o;
	size_t p = 0;
	while (p < s.size() && s[p] == '%') {
		size_t q = p + 1;
		while (q < s.size() && (std::isalnum(static_cast<unsigned char>(s[q])) || s[q] == '_' || s[q] == '.' || s[q] == '-')) {
			++q;
		}
		o.push_back(s.substr(p, q - p));
		p = q;
		if (p + 1 < s.size() && s[p] == ',' && s[p + 1] == ' ' && p + 2 < s.size() && s[p + 2] == '%') {
			p += 2;
		} else {
			break;
		}
	}
	end = p;
	return o;
}

// A value the device can hold: a float, or an i1 as an f32 tensor of 0 and 1. A known i1 gets its tensor here.
bool on_device(const Val &v) {
	return is_float(v.type) || v.type == kPred;
}

Val device_f32(Lowering &L, const Val &v) {
	if (v.type == kPred && v.t == nullptr) {
		Val o = known_val(L, GGML_TYPE_F32, v.shape, v.hv);
		o.type = kPred;
		return as_type(L, o, GGML_TYPE_F32);
	}
	return as_type(L, v, GGML_TYPE_F32);
}

// XLA CPU's select on the reference desk (x86) returns a denormal operand as the zero of its sign (the Mac's XLA
// keeps it; where_runtime measures it). The gather copies bits, so its result is flushed: xla_max(x, x) is x with
// a denormal read as zero.
ggml_tensor *select_flush(Lowering &L, ggml_tensor *t) {
	return ggml_rd_xla_binary_op(L.ctx, t, t, GGML_RD_XLA_MAX);
}

// XLA CPU's maximum, minimum or a compare of two values computed at run time (xla_compare.h).
Val xla_binary(Lowering &L, ggml_rd_xla_binary_kind kind, const Val &a, const Val &b, ggml_type result) {
	const Val fa = device_f32(L, a);
	const Val fb = device_f32(L, b);
	Val o;
	o.t = ggml_rd_xla_binary_op(L.ctx, contiguous(L, fa.t), contiguous(L, fb.t), kind);
	o.shape = a.shape;
	o.type = GGML_TYPE_F32;
	return as_type(L, o, result);
}

// select with a predicate computed at run time: the same gather as a known one, with the row index computed on
// the device as iota + N * (1 - p), exact for N up to 2^24, and converted to i32.
bool select_runtime(Lowering &L, const Val &p, const Val &a, const Val &b, Val &out) {
	const int64_t n = count_of(a.shape);
	if (n > (int64_t(1) << 23)) {
		return L.unsupported("stablehlo.select with a run-time predicate over more than 2^23 elements");
	}
	Val pv = device_f32(L, p);
	ggml_tensor *pt = contiguous(L, pv.t);
	if (p.shape.empty() && !a.shape.empty()) {
		pt = ggml_repeat_4d(L.ctx, ggml_reshape_1d(L.ctx, pt, 1), n, 1, 1, 1);
	}
	pt = ggml_reshape_1d(L.ctx, pt, n);
	std::vector<double> iota(static_cast<size_t>(n));
	for (int64_t i = 0; i < n; ++i) {
		iota[size_t(i)] = double(i);
	}
	const Val base = known_val(L, GGML_TYPE_F32, { n }, iota);
	ggml_tensor *offs = ggml_scale(L.ctx, ggml_sub(L.ctx, filled_like(L, pt, 1.0f), pt), float(n));
	ggml_tensor *idx = ggml_cast(L.ctx, ggml_add(L.ctx, base.t, offs), GGML_TYPE_I32);
	Val fa = device_f32(L, a);
	Val fb = device_f32(L, b);
	ggml_tensor *ra = ggml_reshape_2d(L.ctx, contiguous(L, fa.t), 1, n);
	ggml_tensor *rb = ggml_reshape_2d(L.ctx, contiguous(L, fb.t), 1, n);
	Val o;
	o.t = select_flush(L, ggml_get_rows(L.ctx, ggml_concat(L.ctx, ra, rb, 1), idx));
	o.shape = { n, 1 };
	o.type = GGML_TYPE_F32;
	out = as_type(L, reshaped(L, o, a.shape), a.type);
	return true;
}

// Element-wise ops on values known at compile time that are i32 or i1. Float arithmetic is never folded: it
// runs on the device, so its rounding is the device's.
bool host_binary(Lowering &L, const std::string &op, const Val &a, const Val &b, Val &out) {
	if (!a.known || !b.known) {
		return L.unsupported("stablehlo." + op + " of i32 or i1 values computed at run time");
	}
	if (a.type != b.type) {
		return L.fail("stablehlo." + op + ": operand types differ");
	}
	const bool pred = a.type == kPred;
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const int64_t x = int64_t(a.hv[i]);
		const int64_t y = int64_t(b.hv[i]);
		int64_t r = 0;
		if (op == "add" && !pred) {
			r = x + y;
		} else if (op == "subtract" && !pred) {
			r = x - y;
		} else if (op == "multiply" && !pred) {
			r = int64_t(int32_t(uint32_t(uint64_t(x) * uint64_t(y))));
		} else if (op == "maximum") {
			r = x > y ? x : y;
		} else if (op == "minimum") {
			r = x < y ? x : y;
		} else if (op == "and") {
			r = x & y;
		} else if (op == "or") {
			r = x | y;
		} else if (op == "xor") {
			r = x ^ y;
		} else {
			return L.unsupported("stablehlo." + op + " of " + (pred ? "i1" : "i32") + " values");
		}
		hv[i] = pred ? double(r & 1) : h_wrap32(r);
	}
	out = known_val(L, a.type, a.shape, std::move(hv));
	return true;
}

bool host_unary(Lowering &L, const std::string &op, const Val &a, Val &out) {
	if (!a.known) {
		return L.unsupported("stablehlo." + op + " of i32 or i1 values computed at run time");
	}
	const bool pred = a.type == kPred;
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const int64_t x = int64_t(a.hv[i]);
		if (op == "not") {
			hv[i] = pred ? double(1 - x) : h_wrap32(~x);
		} else if (op == "negate" && !pred) {
			hv[i] = h_wrap32(-x);
		} else if (op == "abs" && !pred) {
			hv[i] = h_wrap32(x < 0 ? -x : x);
		} else {
			return L.unsupported("stablehlo." + op + " of " + (pred ? "i1" : "i32") + " values");
		}
	}
	out = known_val(L, a.type, a.shape, std::move(hv));
	return true;
}

// stablehlo.compare of two values known at compile time.
bool host_compare(Lowering &L, const std::string &dir, const std::string &kind, const Val &a, const Val &b, Val &out) {
	if (a.type != b.type || a.shape != b.shape) {
		return L.fail("stablehlo.compare: operands differ in type or shape");
	}
	if (!a.known || !b.known) {
		if (!on_device(a) || kind == "TOTALORDER") {
			return L.unsupported("stablehlo.compare of i32 values computed at run time, or with TOTALORDER");
		}
		static const char *const kDirs[] = { "GT", "GE", "LT", "LE", "EQ", "NE" };
		static const ggml_rd_xla_binary_kind kKinds[] = { GGML_RD_XLA_GT, GGML_RD_XLA_GE, GGML_RD_XLA_LT, GGML_RD_XLA_LE, GGML_RD_XLA_EQ, GGML_RD_XLA_NE };
		for (int k = 0; k < 6; ++k) {
			if (dir == kDirs[k]) {
				out = xla_binary(L, kKinds[k], a, b, kPred);
				return true;
			}
		}
		return L.fail("stablehlo.compare: unknown direction '" + dir + "'");
	}
	if (kind == "TOTALORDER") {
		return L.unsupported("stablehlo.compare with TOTALORDER");
	}
	const bool uns = kind == "UNSIGNED";
	std::vector<double> hv(a.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		double x = a.hv[i];
		double y = b.hv[i];
		if (uns) {
			x = double(uint32_t(int32_t(x)));
			y = double(uint32_t(int32_t(y)));
		}
		bool r = false;
		if (dir == "EQ") {
			r = x == y;
		} else if (dir == "NE") {
			r = x != y;
		} else if (dir == "LT") {
			r = x < y;
		} else if (dir == "LE") {
			r = x <= y;
		} else if (dir == "GT") {
			r = x > y;
		} else if (dir == "GE") {
			r = x >= y;
		} else {
			return L.fail("stablehlo.compare: unknown direction '" + dir + "'");
		}
		hv[i] = r ? 1.0 : 0.0;
	}
	out = known_val(L, kPred, a.shape, std::move(hv));
	return true;
}

bool op_convert(Lowering &L, const Val &x, const Ty &to, Val &out) {
	if (is_float(x.type) && is_float(to.type)) {
		out = as_type(L, x, to.type);
		return true;
	}
	if (!x.known && x.type == kPred && is_float(to.type)) {
		// A run-time i1 is already an f32 tensor of 0 and 1.
		out = as_type(L, as_type(L, x, GGML_TYPE_F32), to.type);
		return true;
	}
	if (!x.known) {
		return L.unsupported("stablehlo.convert to i32 or i1, or from i32, of a value computed at run time");
	}
	std::vector<double> hv(x.hv.size());
	for (size_t i = 0; i < hv.size(); ++i) {
		const double v = x.hv[i];
		if (to.type == kPred) {
			hv[i] = v != 0.0 ? 1.0 : 0.0;
		} else if (to.type == GGML_TYPE_I32) {
			if (std::isnan(v) || v <= -2147483649.0 || v >= 2147483648.0) {
				return L.unsupported("stablehlo.convert of a NaN or out-of-range float to i32");
			}
			hv[i] = std::trunc(v);
		} else {
			hv[i] = h_round(to.type, v);
		}
	}
	out = known_val(L, to.type, x.shape, std::move(hv));
	return true;
}

bool op_iota(Lowering &L, const std::string &attrs, const Ty &to, Val &out) {
	const size_t d = attrs.find("dim = ");
	if (d == std::string::npos) {
		return L.fail("stablehlo.iota: no dim");
	}
	const int64_t dim = std::strtoll(attrs.c_str() + d + 6, nullptr, 10);
	if (dim < 0 || dim >= int64_t(to.shape.size()) || to.type == kPred) {
		return L.fail("stablehlo.iota: dim is out of range, or the type is i1");
	}
	std::vector<double> hv(size_t(count_of(to.shape)));
	each_index(to.shape, [&](const std::vector<int64_t> &o, size_t flat) { hv[flat] = h_round(to.type, double(o[size_t(dim)])); });
	out = known_val(L, to.type, to.shape, std::move(hv));
	return true;
}

// stablehlo.select with a predicate known at compile time. Float operands that are not both known are picked
// element by element by a gather: both are laid out as rows of one element, b's after a's, and ggml_get_rows
// takes row i or row N + i, a copy that keeps every bit.
bool op_select(Lowering &L, const Val &p, const Val &a, const Val &b, Val &out) {
	if (p.type != kPred || a.type != b.type || a.shape != b.shape || (!p.shape.empty() && p.shape != a.shape)) {
		return L.fail("stablehlo.select: the predicate or the operands do not match");
	}
	if (!p.known) {
		if (!on_device(a)) {
			return L.unsupported("stablehlo.select of i32 values with a predicate computed at run time");
		}
		return select_runtime(L, p, a, b, out);
	}
	const size_t n = size_t(count_of(a.shape));
	bool all_true = true, all_false = true;
	for (double v : p.hv) {
		all_true = all_true && v != 0.0;
		all_false = all_false && v == 0.0;
	}
	if (a.known && b.known) {
		std::vector<double> hv(n);
		for (size_t i = 0; i < n; ++i) {
			hv[i] = (p.shape.empty() ? p.hv[0] : p.hv[i]) != 0.0 ? a.hv[i] : b.hv[i];
		}
		out = known_val(L, a.type, a.shape, std::move(hv));
		return true;
	}
	if (all_true || all_false) {
		out = all_true ? a : b;
		return true;
	}
	if (!on_device(a)) {
		return L.unsupported("stablehlo.select of i32 values computed at run time");
	}
	Val fa = device_f32(L, a);
	Val fb = device_f32(L, b);
	ggml_tensor *ra = ggml_reshape_2d(L.ctx, contiguous(L, fa.t), 1, int64_t(n));
	ggml_tensor *rb = ggml_reshape_2d(L.ctx, contiguous(L, fb.t), 1, int64_t(n));
	ggml_tensor *both = ggml_concat(L.ctx, ra, rb, 1);
	ggml_tensor *idx = ggml_new_tensor_1d(L.ctx, GGML_TYPE_I32, int64_t(n));
	std::vector<uint8_t> bytes(n * 4);
	for (size_t i = 0; i < n; ++i) {
		const int32_t at = int32_t(p.hv[i] != 0.0 ? i : n + i);
		std::memcpy(bytes.data() + i * 4, &at, 4);
	}
	L.consts[idx] = std::move(bytes);
	Val o;
	o.t = select_flush(L, ggml_get_rows(L.ctx, both, idx));
	o.shape = { int64_t(n), 1 };
	o.type = GGML_TYPE_F32;
	out = as_type(L, reshaped(L, o, a.shape), a.type);
	return true;
}

bool statement(Lowering &L, const std::string &line) {
	const size_t eq = line.find(" = ");
	if (line[0] != '%' || eq == std::string::npos) {
		return L.fail("cannot read statement '" + line.substr(0, 60) + "'");
	}
	const std::string name = line.substr(0, eq);
	std::string rest = line.substr(eq + 3);
	if (starts_with(rest, "__alias ")) {
		// The result of an inlined call: the callee's returned value under the caller's name.
		std::map<std::string, Val>::const_iterator it = L.vals.find(rest.substr(8));
		if (it == L.vals.end()) {
			return L.fail("call: " + rest.substr(8) + " is not defined");
		}
		L.vals[name] = it->second;
		return true;
	}
	if (!starts_with(rest, "stablehlo.")) {
		return L.unsupported("'" + rest.substr(0, rest.find(' ')) + "' (only stablehlo ops)");
	}
	const size_t sp = rest.find(' ');
	std::string op = rest.substr(10, sp == std::string::npos ? std::string::npos : sp - 10);
	std::string tail = sp == std::string::npos ? std::string() : rest.substr(sp + 1);
	const size_t paren = op.find('(');
	if (paren != std::string::npos) {
		// "reduce(%x init: %i) ..." has no space before its operand list.
		tail = rest.substr(10 + paren);
		op = op.substr(0, paren);
	}
	if (op == "constant") {
		size_t colon = line.rfind(" : ");
		Ty ty;
		size_t e = 0;
		std::string perr;
		if (colon == std::string::npos || !parse_type_at(line, colon + 3, ty, e, perr)) {
			return L.type_error("stablehlo.constant", perr.empty() ? std::string("no type") : perr);
		}
		Val v;
		if (!op_constant(L, line, ty, v)) {
			return false;
		}
		L.vals[name] = v;
		return true;
	}
	if (op == "reduce" && tail.compare(0, 1, "(") == 0) {
		// "(%x init: %i) applies ..." reads as the operands "%x, %i".
		const size_t close = tail.find(')');
		const size_t init_at = tail.find(" init: ");
		if (close == std::string::npos || init_at == std::string::npos || init_at > close) {
			return L.fail("stablehlo.reduce: cannot read its operands");
		}
		tail = tail.substr(1, init_at - 1) + ", " + tail.substr(init_at + 7, close - init_at - 7) + tail.substr(close + 1);
	}
	std::string direction;
	if (op == "compare") {
		// "compare  GE, %a, %b,  SIGNED : ..." puts the direction first.
		const size_t c = tail.find(',');
		if (c == std::string::npos) {
			return L.fail("stablehlo.compare: no direction");
		}
		direction = trim(tail.substr(0, c));
		tail = trim(tail.substr(c + 1));
	}
	size_t after = 0;
	const std::vector<std::string> ops = operands(tail, after);
	const std::string attrs = tail.substr(after);
	const size_t colon = attrs.rfind(" : ");
	if (colon == std::string::npos) {
		return L.fail("stablehlo." + op + ": no type after ':'");
	}
	const std::string types = attrs.substr(colon + 3);
	Ty to;
	size_t e = 0;
	std::string perr;
	// The result type follows "->"; without one, the last type listed (select lists its predicate's first).
	const size_t arrow = types.rfind("-> ");
	const size_t last = types.rfind("tensor<");
	if (!parse_type_at(types, arrow != std::string::npos ? arrow + 3 : last == std::string::npos ? 0 : last, to, e, perr)) {
		return L.type_error("stablehlo." + op, perr);
	}
	std::vector<Val> in;
	bool all_float = is_float(to.type);
	for (const std::string &o : ops) {
		std::map<std::string, Val>::const_iterator it = L.vals.find(o);
		if (it == L.vals.end()) {
			return L.fail("stablehlo." + op + ": " + o + " is not defined");
		}
		in.push_back(it->second);
		all_float = all_float && is_float(it->second.type);
	}
	Val out;
	bool ok = false;
	bool folds = true; // the op sets out.known itself
	if (op == "add" || op == "subtract" || op == "multiply" || op == "divide" || op == "maximum" || op == "minimum" || op == "and" || op == "or" || op == "xor") {
		if (in.size() != 2) {
			return L.fail("stablehlo." + op + " takes two operands");
		}
		if (in[0].shape != in[1].shape) {
			return L.fail("stablehlo." + op + ": operand shapes differ");
		}
		const bool both_known = in[0].known && in[1].known;
		const bool minmax = op == "maximum" || op == "minimum";
		const bool logic = op == "and" || op == "or" || op == "xor";
		if (all_float && logic) {
			return L.unsupported("stablehlo." + op + " of float values");
		}
		if (all_float && minmax && !(op == "maximum" && (in[0].neg_inf || in[1].neg_inf))) {
			// XLA's maximum and minimum of values computed at run time (a maximum against a constant -inf is
			// softmax's no-op, and op_binary returns the other operand for it, as before).
			const ggml_type rt = in[0].type;
			out = xla_binary(L, op == "maximum" ? GGML_RD_XLA_MAX : GGML_RD_XLA_MIN, in[0], in[1], GGML_TYPE_F32);
			out = as_type(L, out, rt);
			ok = true;
			folds = false;
		} else if (all_float) {
			ok = op_binary(L, op, in[0], in[1], out);
			folds = false;
		} else if (!both_known && in[0].type == kPred && in[1].type == kPred && (logic || minmax)) {
			// i1 values computed at run time are 0 and 1: and is the minimum, or the maximum, xor not-equal.
			const ggml_rd_xla_binary_kind k = op == "and" || op == "minimum" ? GGML_RD_XLA_MIN : op == "xor" ? GGML_RD_XLA_NE : GGML_RD_XLA_MAX;
			out = xla_binary(L, k, in[0], in[1], kPred);
			ok = true;
			folds = false;
		} else {
			ok = host_binary(L, op, in[0], in[1], out);
		}
	} else if (op == "negate" || op == "exponential" || op == "sine" || op == "cosine" || op == "log" || op == "sqrt" || op == "rsqrt" || op == "abs" || op == "floor" || op == "sign" || op == "not") {
		if (in.size() != 1) {
			return L.fail("stablehlo." + op + " takes one operand");
		}
		if (all_float && op != "not") {
			ok = op_unary(L, op, in[0], out);
			folds = false;
		} else if (op == "not" && in[0].type == kPred && !in[0].known) {
			// not of a run-time i1 (0 or 1) is equal-to-zero.
			out = xla_binary(L, GGML_RD_XLA_EQ, in[0], known_val(L, kPred, in[0].shape, std::vector<double>(size_t(count_of(in[0].shape)), 0.0)), kPred);
			ok = true;
			folds = false;
		} else {
			ok = host_unary(L, op, in[0], out);
		}
	} else if (op == "compare") {
		const size_t c = attrs.find(", ");
		const std::string kind = c == std::string::npos || c > colon ? std::string() : trim(attrs.substr(c + 2, colon - c - 2));
		ok = in.size() == 2 ? host_compare(L, direction, kind, in[0], in[1], out) : L.fail("stablehlo.compare takes two operands");
	} else if (op == "select") {
		ok = in.size() == 3 ? op_select(L, in[0], in[1], in[2], out) : L.fail("stablehlo.select takes three operands");
	} else if (op == "iota") {
		ok = in.empty() ? op_iota(L, attrs, to, out) : L.fail("stablehlo.iota takes no operands");
	} else if (op == "convert") {
		ok = in.size() == 1 ? op_convert(L, in[0], to, out) : L.fail("stablehlo.convert takes one operand");
	} else if (op == "reshape") {
		if (in.size() != 1 || count_of(in[0].shape) != count_of(to.shape)) {
			return L.fail("stablehlo.reshape: element counts differ");
		}
		out = reshaped(L, in[0], to.shape);
		ok = true;
	} else if (op == "transpose") {
		std::vector<int64_t> perm;
		if (in.size() != 1 || !int_list(attrs, "dims", 0, perm, nullptr) || perm.size() != in[0].shape.size()) {
			return L.fail("stablehlo.transpose: bad dims");
		}
		out = transposed(L, in[0], perm);
		ok = true;
	} else if (op == "broadcast_in_dim") {
		std::vector<int64_t> dims;
		if (in.size() != 1 || !int_list(attrs, "dims", 0, dims, nullptr)) {
			return L.fail("stablehlo.broadcast_in_dim: bad dims");
		}
		ok = op_broadcast(L, in[0], dims, to, out);
	} else if (op == "slice") {
		ok = in.size() == 1 ? op_slice(L, attrs, in[0], to, out) : L.fail("stablehlo.slice takes one operand");
	} else if (op == "concatenate") {
		ok = op_concatenate(L, attrs, in, out);
	} else if (op == "pad") {
		ok = in.size() == 2 ? op_pad(L, attrs, in[0], in[1], out) : L.fail("stablehlo.pad takes an operand and a padding value");
		folds = false;
	} else if (op == "reduce") {
		ok = in.size() == 2 ? op_reduce(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.reduce takes one operand and an init");
		folds = false;
	} else if (op == "dot_general") {
		ok = in.size() == 2 ? op_dot(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.dot_general takes two operands");
		folds = false;
	} else {
		return L.unsupported("stablehlo." + op);
	}
	if (!ok) {
		return false;
	}
	if (!folds) {
		// A result computed on the device is not known here, even when its operands were.
		out.known = false;
		out.hv.clear();
	}
	if (out.shape != to.shape || out.type != to.type) {
		return L.fail("stablehlo." + op + ": the result is " + std::to_string(out.shape.size()) + "-d of type " + ggml_type_name(out.type)
				+ " but the module declares a different tensor type");
	}
	L.vals[name] = out;
	return true;
}

// Records for every tensor the outputs reach, the parameters first and in order.
bool describe(Lowering &L, const std::vector<ggml_tensor *> &outs, Program &p) {
	std::unordered_map<const ggml_tensor *, int> id;
	std::vector<ggml_tensor *> order;
	for (ggml_tensor *t : L.params) {
		id[t] = int(order.size());
		order.push_back(t);
	}
	std::vector<std::pair<ggml_tensor *, int>> stack;
	for (ggml_tensor *root : outs) {
		stack.emplace_back(root, 0);
		while (!stack.empty()) {
			ggml_tensor *t = stack.back().first;
			const int k = stack.back().second;
			if (id.count(t) != 0) {
				stack.pop_back();
				continue;
			}
			ggml_tensor *dep = nullptr;
			int next = k;
			while (next <= GGML_MAX_SRC && dep == nullptr) {
				ggml_tensor *c = next < GGML_MAX_SRC ? t->src[next] : t->view_src;
				++next;
				if (c != nullptr && c != t && id.count(c) == 0) {
					dep = c;
				}
			}
			if (dep != nullptr) {
				stack.back().second = next;
				stack.emplace_back(dep, 0);
			} else {
				id[t] = int(order.size());
				order.push_back(t);
				stack.pop_back();
			}
		}
	}
	for (size_t i = 0; i < order.size(); ++i) {
		const ggml_tensor *t = order[i];
		TensorRec r;
		r.is_arg = i < L.params.size();
		r.type = int(t->type);
		for (int d = 0; d < GGML_MAX_DIMS; ++d) {
			r.ne[d] = t->ne[d];
			r.nb[d] = uint64_t(t->nb[d]);
		}
		r.op = int(t->op);
		if (!r.is_arg) {
			std::memcpy(r.op_params, t->op_params, sizeof r.op_params);
			for (int s = 0; s < GGML_MAX_SRC; ++s) {
				r.src[s] = t->src[s] == nullptr ? -1 : (t->src[s] == t ? int(i) : id[t->src[s]]);
			}
			if (t->view_src != nullptr) {
				r.view_src = id[t->view_src];
				r.view_offs = uint64_t(t->view_offs);
			}
			if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
				std::unordered_map<const ggml_tensor *, std::vector<uint8_t>>::const_iterator c = L.consts.find(t);
				if (c == L.consts.end()) {
					return L.fail("an operand is neither a parameter nor a constant");
				}
				Program::Const k;
				k.id = int(i);
				k.bytes = c->second;
				p.consts.push_back(std::move(k));
			}
		}
		char nm[32];
		std::snprintf(nm, sizeof nm, "v%zu", i);
		r.name = nm;
		p.t.push_back(std::move(r));
	}
	for (size_t i = 0; i < L.params.size(); ++i) {
		p.args.push_back(int(i));
	}
	for (ggml_tensor *o : outs) {
		p.outs.push_back(id[o]);
	}
	return true;
}

} // namespace

bool is_stablehlo(const std::string &text) {
	return text.find("ggml.arg") == std::string::npos && text.find("func.func") != std::string::npos;
}

namespace {

struct Func {
	std::vector<std::string> params; // %arg names, in order
	std::vector<std::string> body; // statements, then the return line
};

bool ident_char(char c) {
	return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$' || c == '-';
}

// Every %name in line, renamed by f.
template <typename F>
std::string rename(const std::string &line, F f) {
	std::string o;
	size_t i = 0;
	while (i < line.size()) {
		if (line[i] != '%') {
			o += line[i++];
			continue;
		}
		size_t j = i + 1;
		while (j < line.size() && ident_char(line[j])) {
			++j;
		}
		o += f(line.substr(i, j - i));
		i = j;
	}
	return o;
}

// "%r = call @f(%a, %b) : ..." (or func.call): the callee's statements, renamed so they cannot collide, with
// its parameters bound to the caller's operands, then an alias line that gives the returned value the caller's
// name. Calls inside the callee are inlined the same way.
bool inline_call(Lowering &L, const std::map<std::string, Func> &funcs, const std::string &line, int depth, int &counter, std::vector<std::string> &out) {
	const size_t eq = line.find(" = ");
	const size_t at = line.find('@');
	const size_t open = line.find('(', at);
	const size_t close = line.find(')', open);
	if (eq == std::string::npos || at == std::string::npos || open == std::string::npos || close == std::string::npos) {
		return L.fail("call: cannot read '" + line.substr(0, 60) + "'");
	}
	const std::string result = line.substr(0, eq);
	if (result.find(':') != std::string::npos) {
		return L.unsupported("a call with more than one result");
	}
	const std::string callee = line.substr(at + 1, open - at - 1);
	std::map<std::string, Func>::const_iterator f = funcs.find(callee);
	if (f == funcs.end()) {
		return L.fail("call: no function @" + callee);
	}
	if (depth > 32) {
		return L.unsupported("calls nested deeper than 32 (a recursive function)");
	}
	size_t after = 0;
	const std::vector<std::string> args = operands(line.substr(open + 1, close - open - 1), after);
	if (args.size() != f->second.params.size()) {
		return L.fail("call: @" + callee + " takes " + std::to_string(f->second.params.size()) + " operands");
	}
	std::map<std::string, std::string> bound;
	for (size_t i = 0; i < args.size(); ++i) {
		bound[f->second.params[i]] = args[i];
	}
	const std::string prefix = "%i" + std::to_string(counter++) + "_";
	std::function<std::string(const std::string &)> map = [&](const std::string &v) {
		std::map<std::string, std::string>::const_iterator b = bound.find(v);
		return b != bound.end() ? b->second : prefix + v.substr(1);
	};
	for (const std::string &s : f->second.body) {
		const std::string r = rename(s, map);
		if (starts_with(r, "return ")) {
			size_t a2 = 0;
			const std::vector<std::string> ret = operands(r.substr(7), a2);
			if (ret.size() != 1) {
				return L.unsupported("a function that returns other than one value");
			}
			out.push_back(result + " = __alias " + ret[0]);
			return true;
		}
		const size_t e2 = r.find(" = ");
		if (e2 != std::string::npos && (starts_with(r.substr(e2 + 3), "call @") || starts_with(r.substr(e2 + 3), "func.call @"))) {
			if (!inline_call(L, funcs, r, depth + 1, counter, out)) {
				return false;
			}
			continue;
		}
		out.push_back(r);
	}
	return L.fail("call: @" + callee + " has no return");
}

} // namespace

bool lower_stablehlo(const std::string &text, Program &p, std::string &err) {
	Lowering L;
	std::vector<std::string> lines;
	size_t pos = 0;
	while (pos <= text.size()) {
		size_t nl = text.find('\n', pos);
		if (nl == std::string::npos) {
			nl = text.size();
		}
		const std::string l = trim(text.substr(pos, nl - pos));
		if (!l.empty()) {
			lines.push_back(l);
		}
		pos = nl + 1;
	}
	// The functions: @main and the private ones it calls, which are inlined.
	std::map<std::string, Func> funcs;
	std::string main_sig;
	std::string current;
	bool have_main = false;
	for (const std::string &line : lines) {
		if (!L.err.empty()) {
			break;
		}
		if (current.empty()) {
			if (starts_with(line, "module ") || line == "}") {
				continue;
			}
			if (!starts_with(line, "func.func")) {
				L.fail("text outside a function: '" + line.substr(0, 40) + "'");
				break;
			}
			const size_t at = line.find('@');
			const size_t open = line.find('(', at);
			if (at == std::string::npos || open == std::string::npos) {
				L.fail("cannot read '" + line.substr(0, 60) + "'");
				break;
			}
			current = line.substr(at + 1, open - at - 1);
			if (funcs.count(current) != 0) {
				L.fail("function @" + current + " is defined twice");
				break;
			}
			Func &f = funcs[current];
			const size_t close = line.find(") ->", open);
			const std::string sig = close == std::string::npos ? line.substr(open + 1) : line.substr(open + 1, close - open - 1);
			if (current == "main") {
				have_main = true;
				main_sig = sig;
			}
			size_t p2 = 0;
			while ((p2 = sig.find('%', p2)) != std::string::npos) {
				const size_t colon = sig.find(':', p2);
				if (colon == std::string::npos) {
					break;
				}
				f.params.push_back(sig.substr(p2, colon - p2));
				p2 = colon;
			}
			continue;
		}
		if (line == "}") {
			current.clear();
			continue;
		}
		funcs[current].body.push_back(line);
	}
	if (L.err.empty() && !have_main) {
		L.unsupported("a module without a function named @main");
	}
	// main's statements with every call inlined.
	std::vector<std::string> flat;
	int counter = 0;
	if (L.err.empty()) {
		for (const std::string &line : funcs["main"].body) {
			const size_t eq = line.find(" = ");
			if (eq != std::string::npos && (starts_with(line.substr(eq + 3), "call @") || starts_with(line.substr(eq + 3), "func.call @"))) {
				if (!inline_call(L, funcs, line, 0, counter, flat)) {
					break;
				}
				continue;
			}
			flat.push_back(line);
		}
	}
	ggml_init_params ip = { ggml_tensor_overhead() * (flat.size() * 48 + 64) + 4096, nullptr, true };
	L.ctx = ggml_init(ip);
	if (L.ctx == nullptr) {
		err = "ggml_init failed";
		return false;
	}
	// main's parameters, in order.
	size_t p2 = 0;
	while (L.err.empty() && (p2 = main_sig.find("%arg", p2)) != std::string::npos) {
		const size_t colon = main_sig.find(':', p2);
		Ty ty;
		size_t e = 0;
		std::string perr;
		if (colon == std::string::npos || !parse_type_at(main_sig, colon + 2, ty, e, perr)) {
			L.type_error("main parameter", perr.empty() ? std::string("no type") : perr);
			break;
		}
		Val v;
		v.t = new_tensor(L, ty.type, ty.shape);
		v.shape = ty.shape;
		v.type = ty.type;
		L.vals[main_sig.substr(p2, colon - p2)] = v;
		L.params.push_back(v.t);
		p2 = e;
	}
	bool returned = false;
	std::vector<ggml_tensor *> outs;
	for (const std::string &line : flat) {
		if (!L.err.empty()) {
			break;
		}
		if (returned) {
			L.fail("text after the return in @main");
			break;
		}
		if (starts_with(line, "return ") || line == "return") {
			size_t after = 0;
			const std::vector<std::string> ops = operands(line.size() > 7 ? line.substr(7) : std::string(), after);
			const size_t colon = line.find(" : ");
			if (ops.empty() || colon == std::string::npos) {
				L.fail("return without operands or types");
				break;
			}
			for (const std::string &o : ops) {
				std::map<std::string, Val>::const_iterator it = L.vals.find(o);
				if (it == L.vals.end()) {
					L.fail("return: " + o + " is not defined");
					break;
				}
				if (!is_float(it->second.type)) {
					L.unsupported("returning an i32 or i1 value");
					break;
				}
				bool is_param = false;
				for (ggml_tensor *pt : L.params) {
					is_param = is_param || pt == it->second.t;
				}
				if (is_param) {
					L.unsupported("returning a parameter unchanged");
					break;
				}
				outs.push_back(contiguous(L, it->second.t));
			}
			returned = true;
			continue;
		}
		if (!statement(L, line)) {
			break;
		}
	}
	if (L.err.empty() && !returned) {
		L.fail("no return in @main");
	}
	bool ok = L.err.empty() && describe(L, outs, p);
	if (!ok) {
		err = L.err;
	}
	ggml_free(L.ctx);
	return ok;
}

} // namespace pjrt_rd
