#pragma once

#include "denox/memory/container/string.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/Sym.hpp"
#include <utility>

namespace denox::compiler {

struct SymTable {
  memory::vector<std::pair<Sym, memory::string>> symbolNames;
};

} // namespace denox::compiler
