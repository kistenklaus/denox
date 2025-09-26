#pragma once

#include "compiler/ir/impl/TensorId.hpp"
#include "memory/hypergraph/NodeId.hpp"
namespace denox::compiler {

struct MemoryImplicitConcatConstrain {
  // NOTE: Place tensorA and tensorB behind each other in memory.
  TensorId src0;
  TensorId src1;
  TensorId dst;
};

} // namespace denox::compiler
