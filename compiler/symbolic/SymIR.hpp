#pragma once

#include "memory/container/vector.hpp"
#include <cstdint>
namespace denox::compiler {

struct SymIROpCode {

  static constexpr std::size_t OP_VAR = 0;
  static constexpr std::size_t OP_ADD = 1;
  static constexpr std::size_t OP_SUB = 2;
  static constexpr std::size_t OP_MUL = 3;
  static constexpr std::size_t OP_DIV = 4;
  static constexpr std::size_t OP_MOD = 5;
  static constexpr std::size_t OP_MIN = 6;
  static constexpr std::size_t OP_MAX = 7;

  std::uint8_t lhsIsConstant : 1;
  std::uint8_t rhsIsConstant : 1;
  std::uint8_t op : 8;
};

struct SymIROp {
  SymIROpCode opcode;
  std::int64_t lhs;
  std::int64_t rhs;
};

struct SymIR {
  memory::vector<SymIROp> ops;
};

} // namespace denox::compiler
