#pragma once

#include "compiler/ir/TensorInstance.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "model/ComputeOp.hpp"

namespace denox::compiler {

struct SpecModel {
  using Graph = memory::LinkedGraph<TensorInstance, ComputeOp>;
  Graph graph;
  Graph::NodeHandle input;
  Graph::NodeHandle output;
};

} // namespace denox::compiler
