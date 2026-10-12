// The web build's stand-in for godot-sandbox's api.hpp: ggml-rd needs only the
// RID handle, which is an integer here (a slot in the page's WebGPU object
// table, web/rd_compute_web.cpp).
#pragma once

#include <cstdint>

struct RID {
	int64_t index = 0;
	bool is_valid() const { return index != 0; }
};
