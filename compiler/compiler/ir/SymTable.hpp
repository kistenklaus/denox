#pragma once

#include "memory/container/string.hpp"
#include "memory/container/vector.hpp"
#include "symbolic/Sym.hpp"
#include <utility>

namespace denox::compiler {

struct SymTable {
  memory::vector<std::pair<Sym, memory::string>> symbolNames;
};

} // namespace denox::compiler
