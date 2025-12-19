#pragma once

#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
namespace denox::compiler {

struct AdjModel {
  using Graph = memory::AdjGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  memory::NodeId input;
  memory::NodeId output;
};

} // namespace denox::compiler
