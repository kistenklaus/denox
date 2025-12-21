#pragma once

#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"

namespace denox::compiler {

struct SpecModel {
  using Graph = memory::LinkedGraph<TensorInstance, ComputeOp>;
  Graph graph;
  Graph::NodeHandle input;
  Graph::NodeHandle output;
};

} // namespace denox::compiler
