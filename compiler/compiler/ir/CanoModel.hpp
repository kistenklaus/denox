#pragma once

#include "memory/hypergraph/LinkedGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include "symbolic/SymGraph.hpp"
namespace denox::compiler {

struct CanoModel {
  using Graph = memory::LinkedGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  Graph::NodeHandle input;
  Graph::NodeHandle output;
  SymGraph symGraph;
};

} // namespace denox::compiler
