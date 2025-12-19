#pragma once

#include "denox/memory/container/vector.hpp"
#include <cstdint>
namespace denox::algorithm {

struct BinaryOp {
  bool lhsIntermediate;
  std::uint32_t lhs; 
  bool rhsIntermediate;
  std::uint32_t rhs;
};

struct BinaryOpPermutation {
  memory::vector<BinaryOp> ops;
};

memory::vector<BinaryOpPermutation> binary_op_permutation(std::uint32_t n);

} // namespace denox::algorithm
