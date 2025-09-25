#pragma once

#include "compiler/ir/SpecModel.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "model/ComputeOp.hpp"

namespace denox::compiler {

struct OpModel {
  using Graph = memory::ConstGraph<TensorInstance, ComputeOp>;
  Graph graph;
  memory::NodeId input;
  memory::NodeId output;
};

} // namespace denox::compiler
