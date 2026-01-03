#pragma once

#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymIR.hpp"

namespace denox::compiler {

struct SymProgram {
  memory::vector<NamedValue> namedValues;
  SymIR ir;
};

} // namespace denox::compiler
