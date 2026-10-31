#include "stablehlo.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

#include "ggml.h"

namespace pjrt_rd {

namespace {

struct Val {
	ggml_tensor *t = nullptr;
	std::vector<int64_t> shape; // row-major, as StableHLO writes it
	ggml_type type = GGML_TYPE_F32;
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
	} else {
		err = "unsupported: element type '" + elem + "' (f32 and bf16 only)";
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
	if (out.shape.size() > GGML_MAX_DIMS) {
		err = "unsupported: rank above 4";
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

void ne_of(const std::vector<int64_t> &shape, int64_t ne[GGML_MAX_DIMS]) {
	for (int i = 0; i < GGML_MAX_DIMS; ++i) {
		ne[i] = 1;
	}
	const size_t r = shape.size();
	for (size_t i = 0; i < r; ++i) {
		ne[i] = shape[r - 1 - i];
	}
}

ggml_tensor *new_tensor(Lowering &L, ggml_type type, const std::vector<int64_t> &shape) {
	int64_t ne[GGML_MAX_DIMS];
	ne_of(shape, ne);
	return ggml_new_tensor_4d(L.ctx, type, ne[0], ne[1], ne[2], ne[3]);
}

ggml_tensor *contiguous(Lowering &L, ggml_tensor *t) {
	return ggml_is_contiguous(t) ? t : ggml_cont(L.ctx, t);
}

Val reshaped(Lowering &L, const Val &v, const std::vector<int64_t> &shape) {
	int64_t ne[GGML_MAX_DIMS];
	ne_of(shape, ne);
	Val o;
	o.t = ggml_reshape_4d(L.ctx, contiguous(L, v.t), ne[0], ne[1], ne[2], ne[3]);
	o.shape = shape;
	o.type = v.type;
	return o;
}

Val as_type(Lowering &L, const Val &v, ggml_type type) {
	if (v.type == type) {
		return v;
	}
	Val o = v;
	o.t = ggml_cast(L.ctx, v.t, type);
	o.type = type;
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
	for (int i = 0; i < r; ++i) {
		o.shape.push_back(v.shape[size_t(perm[size_t(i)])]);
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

bool op_unary(Lowering &L, const std::string &op, const Val &a, Val &out) {
	Val f = as_type(L, a, GGML_TYPE_F32);
	Val o = f;
	if (op == "negate") {
		o.t = ggml_neg(L.ctx, f.t);
	} else if (op == "exponential") {
		o.t = ggml_exp(L.ctx, f.t);
	} else if (op == "log") {
		o.t = ggml_log(L.ctx, f.t);
	} else if (op == "sqrt") {
		o.t = ggml_sqrt(L.ctx, f.t);
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
		o.t = ggml_div(L.ctx, fa.t, fb.t);
	} else {
		return L.unsupported("stablehlo." + op);
	}
	out = as_type(L, o, a.type);
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
	Val r = reshaped(L, x, mid);
	if (mid == to.shape) {
		out = r;
		return true;
	}
	Val o;
	int64_t ne[GGML_MAX_DIMS];
	ne_of(to.shape, ne);
	o.t = ggml_repeat_4d(L.ctx, r.t, ne[0], ne[1], ne[2], ne[3]);
	o.shape = to.shape;
	o.type = x.type;
	out = o;
	return true;
}

bool op_dot(Lowering &L, const std::string &attrs, const Val &lhs, const Val &rhs, const Ty &to, Val &out) {
	std::vector<int64_t> lb, rb, lc, rc;
	int_list_pair(attrs, "batching_dims", lb, rb);
	if (!int_list_pair(attrs, "contracting_dims", lc, rc) || lc.size() != 1 || rc.size() != 1 || lb.size() != rb.size()) {
		return L.unsupported("stablehlo.dot_general without exactly one contracting dimension per side");
	}
	if (lb.size() > 2) {
		return L.unsupported("stablehlo.dot_general with more than two batch dimensions");
	}
	std::vector<int64_t> lfree, rfree;
	for (int64_t d = 0; d < int64_t(lhs.shape.size()); ++d) {
		bool used = d == lc[0];
		for (int64_t b : lb) {
			used = used || b == d;
		}
		if (!used) {
			lfree.push_back(d);
		}
	}
	for (int64_t d = 0; d < int64_t(rhs.shape.size()); ++d) {
		bool used = d == rc[0];
		for (int64_t b : rb) {
			used = used || b == d;
		}
		if (!used) {
			rfree.push_back(d);
		}
	}
	if (lfree.size() > 1 || rfree.size() > 1) {
		return L.unsupported("stablehlo.dot_general with more than one free dimension on a side");
	}
	if (lhs.shape[size_t(lc[0])] != rhs.shape[size_t(rc[0])]) {
		return L.fail("stablehlo.dot_general: contracted dimensions differ");
	}
	for (size_t i = 0; i < lb.size(); ++i) {
		if (lhs.shape[size_t(lb[i])] != rhs.shape[size_t(rb[i])]) {
			return L.fail("stablehlo.dot_general: batch dimensions differ");
		}
	}
	std::vector<int64_t> lperm = lb;
	lperm.insert(lperm.end(), lfree.begin(), lfree.end());
	lperm.push_back(lc[0]);
	std::vector<int64_t> rperm = rb;
	rperm.insert(rperm.end(), rfree.begin(), rfree.end());
	rperm.push_back(rc[0]);
	Val l = transposed(L, lhs, lperm);
	Val r = transposed(L, rhs, rperm);
	if (lfree.empty()) {
		std::vector<int64_t> s = l.shape;
		s.insert(s.end() - 1, 1);
		l = reshaped(L, l, s);
	}
	if (rfree.empty()) {
		std::vector<int64_t> s = r.shape;
		s.insert(s.end() - 1, 1);
		r = reshaped(L, r, s);
	}
	l = as_type(L, l, GGML_TYPE_F32);
	Val o;
	o.t = ggml_mul_mat(L.ctx, contiguous(L, r.t), contiguous(L, l.t));
	o.type = GGML_TYPE_F32;
	for (size_t i = 0; i < lb.size(); ++i) {
		o.shape.push_back(lhs.shape[size_t(lb[i])]);
	}
	o.shape.push_back(lfree.empty() ? 1 : lhs.shape[size_t(lfree[0])]);
	o.shape.push_back(rfree.empty() ? 1 : rhs.shape[size_t(rfree[0])]);
	std::vector<int64_t> want;
	for (size_t i = 0; i < lb.size(); ++i) {
		want.push_back(lhs.shape[size_t(lb[i])]);
	}
	if (!lfree.empty()) {
		want.push_back(lhs.shape[size_t(lfree[0])]);
	}
	if (!rfree.empty()) {
		want.push_back(rhs.shape[size_t(rfree[0])]);
	}
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
	if (body.find('"') != std::string::npos || body.find("0x") != std::string::npos || body.find("inf") != std::string::npos || body.find("nan") != std::string::npos) {
		return L.unsupported("stablehlo.constant with a hex, string, inf or nan literal");
	}
	std::vector<float> values;
	size_t p = 0;
	while (p < body.size()) {
		const char c = body[p];
		if (c == '[' || c == ']' || c == ',' || c == ' ') {
			++p;
			continue;
		}
		char *end = nullptr;
		const double v = std::strtod(body.c_str() + p, &end);
		if (end == body.c_str() + p) {
			return L.fail("stablehlo.constant: cannot read '" + body.substr(p, 12) + "'");
		}
		values.push_back(float(v));
		p = size_t(end - body.c_str());
	}
	const int64_t n = count_of(ty.shape);
	if (values.size() != 1 && int64_t(values.size()) != n) {
		return L.fail("stablehlo.constant: " + std::to_string(values.size()) + " values for " + std::to_string(n) + " elements");
	}
	Val v;
	v.t = new_tensor(L, ty.type, ty.shape);
	v.shape = ty.shape;
	v.type = ty.type;
	const size_t esz = ty.type == GGML_TYPE_F32 ? 4 : 2;
	std::vector<uint8_t> bytes(size_t(n) * esz);
	for (int64_t i = 0; i < n; ++i) {
		const float f = values.size() == 1 ? values[0] : values[size_t(i)];
		if (ty.type == GGML_TYPE_F32) {
			std::memcpy(bytes.data() + size_t(i) * 4, &f, 4);
		} else {
			const ggml_bf16_t b = ggml_fp32_to_bf16(f);
			std::memcpy(bytes.data() + size_t(i) * 2, &b, 2);
		}
	}
	L.consts[v.t] = std::move(bytes);
	out = v;
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

bool statement(Lowering &L, const std::string &line) {
	const size_t eq = line.find(" = ");
	if (line[0] != '%' || eq == std::string::npos) {
		return L.fail("cannot read statement '" + line.substr(0, 60) + "'");
	}
	const std::string name = line.substr(0, eq);
	std::string rest = line.substr(eq + 3);
	if (!starts_with(rest, "stablehlo.")) {
		return L.unsupported("'" + rest.substr(0, rest.find(' ')) + "' (only stablehlo ops)");
	}
	const size_t sp = rest.find(' ');
	const std::string op = rest.substr(10, sp == std::string::npos ? std::string::npos : sp - 10);
	std::string tail = sp == std::string::npos ? std::string() : rest.substr(sp + 1);
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
	const size_t arrow = types.rfind("-> ");
	if (!parse_type_at(types, arrow == std::string::npos ? 0 : arrow + 3, to, e, perr)) {
		return L.type_error("stablehlo." + op, perr);
	}
	std::vector<Val> in;
	for (const std::string &o : ops) {
		std::map<std::string, Val>::const_iterator it = L.vals.find(o);
		if (it == L.vals.end()) {
			return L.fail("stablehlo." + op + ": " + o + " is not defined");
		}
		in.push_back(it->second);
	}
	Val out;
	bool ok = false;
	if (op == "add" || op == "subtract" || op == "multiply" || op == "divide") {
		ok = in.size() == 2 ? op_binary(L, op, in[0], in[1], out) : L.fail("stablehlo." + op + " takes two operands");
	} else if (op == "negate" || op == "exponential" || op == "log" || op == "sqrt") {
		ok = in.size() == 1 ? op_unary(L, op, in[0], out) : L.fail("stablehlo." + op + " takes one operand");
	} else if (op == "convert") {
		if (in.size() != 1) {
			return L.fail("stablehlo.convert takes one operand");
		}
		out = as_type(L, in[0], to.type);
		ok = true;
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
	} else if (op == "dot_general") {
		ok = in.size() == 2 ? op_dot(L, attrs, in[0], in[1], to, out) : L.fail("stablehlo.dot_general takes two operands");
	} else {
		return L.unsupported("stablehlo." + op);
	}
	if (!ok) {
		return false;
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
	ggml_init_params ip = { ggml_tensor_overhead() * (lines.size() * 24 + 64) + 4096, nullptr, true };
	L.ctx = ggml_init(ip);
	if (L.ctx == nullptr) {
		err = "ggml_init failed";
		return false;
	}
	bool in_main = false;
	bool returned = false;
	std::vector<ggml_tensor *> outs;
	for (const std::string &line : lines) {
		if (starts_with(line, "module ") || line == "}") {
			if (line == "}" && in_main) {
				in_main = false;
			}
			continue;
		}
		if (starts_with(line, "func.func")) {
			if (in_main || returned || line.find("@main(") == std::string::npos) {
				L.unsupported("a module with more than one function, or none named @main");
				break;
			}
			in_main = true;
			const size_t open = line.find("@main(") + 6;
			const size_t close = line.find(") ->", open);
			const std::string sig = close == std::string::npos ? line.substr(open) : line.substr(open, close - open);
			size_t p2 = 0;
			while ((p2 = sig.find("%arg", p2)) != std::string::npos) {
				const size_t colon = sig.find(':', p2);
				Ty ty;
				size_t e = 0;
				std::string perr;
				if (colon == std::string::npos || !parse_type_at(sig, colon + 2, ty, e, perr)) {
					L.type_error("main parameter", perr.empty() ? std::string("no type") : perr);
					break;
				}
				Val v;
				v.t = new_tensor(L, ty.type, ty.shape);
				v.shape = ty.shape;
				v.type = ty.type;
				L.vals[sig.substr(p2, colon - p2)] = v;
				L.params.push_back(v.t);
				p2 = e;
			}
			if (!L.err.empty()) {
				break;
			}
			continue;
		}
		if (!in_main) {
			L.fail("text outside func @main: '" + line.substr(0, 40) + "'");
			break;
		}
		if (starts_with(line, "return ") || line == "return") {
			size_t after = 0;
			const std::vector<std::string> ops = operands(line.substr(7), after);
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
			if (!L.err.empty()) {
				break;
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
