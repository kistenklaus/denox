#pragma once

#include "memory/container/vector.hpp"
#include <cstdint>
namespace denox::compiler {

enum class SymIROpCode {
  Add_SS,
  Add_SC,

  Sub_SS,
  Sub_SC,
  Sub_CS,

  Mul_SS,
  Mul_SC,

  Div_SS,
  Div_SC,
  Div_CS,

  Mod_SS,
  Mod_SC,
  Mod_CS,

  Min_SS,
  Min_SC,

  Max_SS,
  Max_SC,
};

struct SymIROp {
  SymIROpCode opcode;
  std::int64_t lhs;
  std::int64_t rhs;
};

struct SymIR {
  std::size_t varCount;
  memory::vector<SymIROp> ops;
};

} // namespace denox::compiler
