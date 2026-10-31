#include "program.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace pjrt_rd {

namespace {

void put_ints(std::string &o, const char *key, const int64_t *v, int n) {
	char b[32];
	o += key;
	o += " = [";
	for (int i = 0; i < n; ++i) {
		std::snprintf(b, sizeof b, "%s%lld", i ? ", " : "", (long long)v[i]);
		o += b;
	}
	o += "]";
}

std::string clean(const char *s) {
	std::string r(s);
	for (char &c : r) {
		if (c == '"' || c == '\n' || c == '\\') {
			c = '_';
		}
	}
	return r;
}

} // namespace

std::string serialize(const std::vector<ggml_tensor *> &outs, std::vector<ggml_tensor *> &args) {
	std::unordered_map<const ggml_tensor *, int> id;
	std::vector<ggml_tensor *> order;
	std::vector<std::pair<ggml_tensor *, int>> stack;
	args.clear();
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
				for (size_t d = 0; d < stack.size(); ++d) {
					if (stack[d].first == dep) {
						std::printf("serialize: a cycle through %s '%s' at depth %zu\n", ggml_op_name(dep->op), dep->name, d);
						std::fflush(stdout);
						return std::string();
					}
				}
				stack.back().second = next;
				stack.emplace_back(dep, 0);
			} else {
				id[t] = int(order.size());
				order.push_back(t);
				stack.pop_back();
			}
		}
	}
	std::string o = "module @ggml_graph attributes {ggml.version = 1} {\n  func.func public @main() {\n";
	char b[64];
	for (size_t i = 0; i < order.size(); ++i) {
		const ggml_tensor *t = order[i];
		const bool arg = t->op == GGML_OP_NONE && t->view_src == nullptr;
		if (arg) {
			args.push_back(order[i]);
		}
		std::snprintf(b, sizeof b, "    %%%zu = ", i);
		o += b;
		if (arg) {
			o += "\"ggml.arg\"() {";
		} else {
			o += "\"stablehlo.custom_call\"(";
			bool first = true;
			for (int s = 0; s < GGML_MAX_SRC; ++s) {
				if (t->src[s] != nullptr) {
					std::snprintf(b, sizeof b, "%s%%%d", first ? "" : ", ", t->src[s] == t ? int(i) : id[t->src[s]]);
					o += b;
					first = false;
				}
			}
			o += ") {call_target_name = \"ggml.";
			o += ggml_op_name(t->op);
			o += "\", ";
		}
		o += "type = \"";
		o += ggml_type_name(t->type);
		o += "\", ";
		put_ints(o, "ne", t->ne, GGML_MAX_DIMS);
		o += ", nb = [";
		for (int d = 0; d < GGML_MAX_DIMS; ++d) {
			std::snprintf(b, sizeof b, "%s%llu", d ? ", " : "", (unsigned long long)t->nb[d]);
			o += b;
		}
		o += "]";
		if (!arg) {
			o += ", op_params = [";
			for (size_t p = 0; p < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++p) {
				std::snprintf(b, sizeof b, "%s%d", p ? ", " : "", (int)t->op_params[p]);
				o += b;
			}
			std::snprintf(b, sizeof b, "], flags = %d", (int)t->flags);
			o += b;
			const int vs = t->view_src ? id[t->view_src] : -1;
			std::snprintf(b, sizeof b, ", view_src = %d, view_offs = %llu", vs, (unsigned long long)t->view_offs);
			o += b;
			o += ", src_ids = [";
			bool first = true;
			for (int s = 0; s < GGML_MAX_SRC; ++s) {
				std::snprintf(b, sizeof b, "%s%d", first ? "" : ", ", t->src[s] == t ? int(i) : (t->src[s] ? id[t->src[s]] : -1));
				o += b;
				first = false;
			}
			o += "]";
		}
		o += ", name = \"" + clean(t->name) + "\"}\n";
	}
	o += "    \"func.return\"(";
	for (size_t i = 0; i < outs.size(); ++i) {
		std::snprintf(b, sizeof b, "%s%%%d", i ? ", " : "", id[outs[i]]);
		o += b;
	}
	o += ")\n  }\n}\n";
	return o;
}

namespace {

struct Cursor {
	const char *p;
	const char *end;
	void ws() {
		while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) {
			++p;
		}
	}
};

bool read_int(Cursor &c, long long &v) {
	c.ws();
	char *e = nullptr;
	v = std::strtoll(c.p, &e, 10);
	if (e == c.p) {
		return false;
	}
	c.p = e;
	return true;
}

bool read_uint(Cursor &c, unsigned long long &v) {
	c.ws();
	char *e = nullptr;
	v = std::strtoull(c.p, &e, 10);
	if (e == c.p) {
		return false;
	}
	c.p = e;
	return true;
}

int op_by_name(const std::string &n) {
	for (int i = 0; i < GGML_OP_COUNT; ++i) {
		if (n == ggml_op_name(ggml_op(i))) {
			return i;
		}
	}
	return -1;
}

int type_by_name(const std::string &n) {
	for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
		if (n == ggml_type_name(ggml_type(i))) {
			return i;
		}
	}
	return -1;
}

bool parse_line(const std::string &line, Program &p, std::string &err) {
	Cursor c{ line.data(), line.data() + line.size() };
	c.ws();
	if (c.p >= c.end) {
		return true;
	}
	if (*c.p == '"') {
		const std::string tag = "\"func.return\"(";
		if (line.compare(size_t(c.p - line.data()), tag.size(), tag) != 0) {
			return true;
		}
		c.p += tag.size();
		while (c.p < c.end && *c.p != ')') {
			c.ws();
			if (*c.p == '%') {
				++c.p;
				long long v = 0;
				if (!read_int(c, v)) {
					err = "bad return operand";
					return false;
				}
				p.outs.push_back(int(v));
			}
			c.ws();
			if (c.p < c.end && *c.p == ',') {
				++c.p;
			}
		}
		return true;
	}
	if (*c.p != '%') {
		return true;
	}
	++c.p;
	long long idv = 0;
	if (!read_int(c, idv) || idv != (long long)p.t.size()) {
		err = "result ids must be dense and in order";
		return false;
	}
	const char *eq = std::strstr(c.p, "= \"");
	if (eq == nullptr) {
		err = "no op after the result id";
		return false;
	}
	c.p = eq + 3;
	const char *q = std::strchr(c.p, '"');
	if (q == nullptr) {
		err = "unterminated op name";
		return false;
	}
	const std::string opname(c.p, q);
	c.p = q + 1;
	TensorRec r;
	r.is_arg = opname == "ggml.arg";
	if (!r.is_arg && opname != "stablehlo.custom_call") {
		err = "unsupported op " + opname;
		return false;
	}
	c.ws();
	if (c.p >= c.end || *c.p != '(') {
		err = "no operand list";
		return false;
	}
	++c.p;
	std::vector<int> operands;
	while (c.p < c.end && *c.p != ')') {
		c.ws();
		if (*c.p == '%') {
			++c.p;
			long long v = 0;
			if (!read_int(c, v) || v < 0 || v > idv) {
				err = "operand refers forward";
				return false;
			}
			operands.push_back(int(v));
		}
		c.ws();
		if (c.p < c.end && *c.p == ',') {
			++c.p;
		}
	}
	++c.p;
	c.ws();
	if (c.p >= c.end || *c.p != '{') {
		err = "no attribute dictionary";
		return false;
	}
	++c.p;
	std::string target;
	bool have_type = false, have_ne = false, have_nb = false;
	while (c.p < c.end && *c.p != '}') {
		c.ws();
		const char *k = c.p;
		while (c.p < c.end && *c.p != ' ' && *c.p != '=') {
			++c.p;
		}
		const std::string key(k, c.p);
		c.ws();
		if (c.p >= c.end || *c.p != '=') {
			err = "attribute " + key + " has no value";
			return false;
		}
		++c.p;
		c.ws();
		std::string sval;
		std::vector<long long> ivals;
		std::vector<unsigned long long> uvals;
		bool is_str = false, is_list = false;
		if (*c.p == '"') {
			is_str = true;
			const char *e = std::strchr(c.p + 1, '"');
			if (e == nullptr) {
				err = "unterminated string";
				return false;
			}
			sval.assign(c.p + 1, e);
			c.p = e + 1;
		} else if (*c.p == '[') {
			is_list = true;
			++c.p;
			while (c.p < c.end && *c.p != ']') {
				unsigned long long u = 0;
				c.ws();
				if (*c.p == '-') {
					long long v = 0;
					if (!read_int(c, v)) {
						err = "bad list element";
						return false;
					}
					ivals.push_back(v);
					uvals.push_back((unsigned long long)v);
				} else {
					if (!read_uint(c, u)) {
						err = "bad list element";
						return false;
					}
					ivals.push_back((long long)u);
					uvals.push_back(u);
				}
				c.ws();
				if (c.p < c.end && *c.p == ',') {
					++c.p;
				}
			}
			++c.p;
		} else {
			long long v = 0;
			if (*c.p == '-') {
				if (!read_int(c, v)) {
					err = "bad number";
					return false;
				}
				ivals.assign(1, v);
				uvals.assign(1, (unsigned long long)v);
			} else {
				unsigned long long u = 0;
				if (!read_uint(c, u)) {
					err = "bad number for " + key;
					return false;
				}
				ivals.assign(1, (long long)u);
				uvals.assign(1, u);
			}
		}
		c.ws();
		if (c.p < c.end && *c.p == ',') {
			++c.p;
		}
		if (key == "call_target_name" && is_str) {
			target = sval;
		} else if (key == "type" && is_str) {
			r.type = type_by_name(sval);
			have_type = r.type >= 0;
		} else if (key == "name" && is_str) {
			r.name = sval;
		} else if (key == "ne" && is_list && ivals.size() == GGML_MAX_DIMS) {
			for (int d = 0; d < GGML_MAX_DIMS; ++d) {
				r.ne[d] = ivals[d];
			}
			have_ne = true;
		} else if (key == "nb" && is_list && uvals.size() == GGML_MAX_DIMS) {
			for (int d = 0; d < GGML_MAX_DIMS; ++d) {
				r.nb[d] = uvals[d];
			}
			have_nb = true;
		} else if (key == "op_params" && is_list && ivals.size() == GGML_MAX_OP_PARAMS / sizeof(int32_t)) {
			for (size_t i = 0; i < ivals.size(); ++i) {
				r.op_params[i] = int32_t(ivals[i]);
			}
		} else if (key == "src_ids" && is_list && ivals.size() == GGML_MAX_SRC) {
			for (int i = 0; i < GGML_MAX_SRC; ++i) {
				r.src[i] = int(ivals[i]);
			}
		} else if (key == "flags" && !is_list && !is_str) {
			r.flags = int32_t(ivals[0]);
		} else if (key == "view_src" && !is_list && !is_str) {
			r.view_src = int(ivals[0]);
		} else if (key == "view_offs" && !is_list && !is_str) {
			r.view_offs = uvals[0];
		}
	}
	if (!have_type || !have_ne || !have_nb) {
		err = "result " + std::to_string(idv) + " lacks type, ne or nb";
		return false;
	}
	if (!r.is_arg) {
		if (target.compare(0, 5, "ggml.") != 0) {
			err = "custom_call target is not a ggml op: " + target;
			return false;
		}
		r.op = op_by_name(target.substr(5));
		if (r.op < 0) {
			err = "unknown ggml op " + target;
			return false;
		}
		for (int i = 0; i < GGML_MAX_SRC; ++i) {
			if (r.src[i] > idv || r.src[i] < -1) {
				err = "src id out of range";
				return false;
			}
		}
		if (r.view_src >= idv || r.view_src < -1) {
			err = "view_src id out of range";
			return false;
		}
		int n = 0;
		for (int i = 0; i < GGML_MAX_SRC; ++i) {
			n += r.src[i] >= 0;
		}
		if (n != int(operands.size())) {
			err = "operand count differs from src_ids";
			return false;
		}
	} else {
		p.args.push_back(int(idv));
	}
	p.t.push_back(r);
	return true;
}

} // namespace

bool parse(const std::string &text, Program &p, std::string &err) {
	p = Program();
	if (text.compare(0, 20, "module @ggml_graph a") != 0) {
		err = "not a ggml graph module";
		return false;
	}
	size_t pos = 0;
	while (pos < text.size()) {
		size_t nl = text.find('\n', pos);
		if (nl == std::string::npos) {
			nl = text.size();
		}
		if (!parse_line(text.substr(pos, nl - pos), p, err)) {
			return false;
		}
		pos = nl + 1;
	}
	if (p.outs.empty()) {
		err = "the module returns nothing";
		return false;
	}
	for (int o : p.outs) {
		if (o < 0 || size_t(o) >= p.t.size()) {
			err = "return operand out of range";
			return false;
		}
	}
	return true;
}

} // namespace pjrt_rd
