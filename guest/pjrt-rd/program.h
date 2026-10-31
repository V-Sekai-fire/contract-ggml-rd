// A ggml graph carried as a StableHLO-shaped module: each tensor is one op line,
// "ggml.arg" for a leaf the caller supplies and a stablehlo.custom_call to
// "ggml.<OP>" for a node or view. The ops keep their ggml meaning (type, ne, nb,
// op_params, view_src, view_offs), so the plugin hands them to ggml-rd unchanged.
// A result may name itself as an operand: ggml_cast makes its CPY the destination of its own copy.
// This is a subset of the syntax, not a StableHLO lowering.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace pjrt_rd {

struct TensorRec {
	bool is_arg = false;
	int type = 0;
	int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
	uint64_t nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
	int op = 0;
	int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)] = {};
	int32_t flags = 0;
	int src[GGML_MAX_SRC];
	int view_src = -1;
	uint64_t view_offs = 0;
	std::string name;
	TensorRec() {
		for (int i = 0; i < GGML_MAX_SRC; ++i) {
			src[i] = -1;
		}
	}
};

struct Program {
	struct Const {
		int id = -1; // the record this leaf's bytes belong to
		std::vector<uint8_t> bytes;
	};
	std::vector<TensorRec> t;
	std::vector<Const> consts; // leaves that are not arguments: data the module itself carries
	std::vector<int> args; // ids of the ggml.arg records, in argument order
	std::vector<int> outs; // ids of the results, in result order
};

// Everything the outputs depend on, dependencies first. `args` receives the leaf
// tensors in argument order.
std::string serialize(const std::vector<ggml_tensor *> &outs, std::vector<ggml_tensor *> &args);

bool parse(const std::string &text, Program &p, std::string &err);

} // namespace pjrt_rd
