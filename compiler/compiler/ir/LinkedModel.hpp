#pragma once

#include "memory/hypergraph/LinkedGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
namespace denox::compiler {

struct LinkedModel {
  using Graph = memory::LinkedGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  Graph::NodeHandle input;
  Graph::NodeHandle output;
};

} // namespace denox::compiler
