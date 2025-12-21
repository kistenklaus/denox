#pragma once

#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct CanoModel {
  using Graph = memory::LinkedGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  std::vector<Graph::NodeHandle> inputs;
  std::vector<Graph::NodeHandle> outputs;
  SymGraph symGraph;
};

} // namespace denox::compiler
