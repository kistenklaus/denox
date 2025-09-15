#pragma once

#include "memory/hypergraph/ConstGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
namespace denox::compiler {

struct ConstModel {
  using Graph = memory::ConstGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  memory::NodeId input;
  memory::NodeId output;
};

}
