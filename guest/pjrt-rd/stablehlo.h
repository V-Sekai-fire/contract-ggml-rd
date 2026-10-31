// StableHLO text to a Program: the module's single func @main becomes a ggml graph, which is then
// described record by record. Row-major shapes become ggml ne by reversing the dimensions.
#pragma once

#include <string>

#include "program.h"

namespace pjrt_rd {

// True when the text is a StableHLO module and not the ggml-carrying one program.h writes.
bool is_stablehlo(const std::string &text);

// err starts with "unsupported" when the module is valid StableHLO outside the supported ops.
bool lower_stablehlo(const std::string &text, Program &p, std::string &err);

} // namespace pjrt_rd
