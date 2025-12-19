#pragma once

#include "memory/container/vector.hpp"
#include "memory/dtype/dtype.hpp"
#include <cstdint>
namespace denox::compiler {

struct CoopmatShape {
  // A is MxK
  // B is KxN
  // C is MxN
  // C = A * B + C (multiply accumulate)
  std::uint32_t M;
  std::uint32_t N;
  std::uint32_t K;
  memory::Dtype atype; 
  memory::Dtype btype;
  memory::Dtype ctype;
  memory::Dtype acctype;
  bool saturatingAccumulation;
  bool subgroupScope;
};

struct CoopmatProperties {
  bool supported;
  memory::vector<CoopmatShape> shapes;
};

} // namespace denox::compiler
