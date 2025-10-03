#pragma once

#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include "model/DynamicInputExtent.hpp"
#include "model/ModelMeta.hpp"
#include "symbolic/SymGraph.hpp"

namespace denox::compiler::details::model {

struct ModelControlBlock {
  static constexpr memory::NodeId NullNode{static_cast<std::size_t>(-1)};
  ModelControlBlock()
      : input(NullNode), output(NullNode), hypergraph(), symGraph() {}

  NamedExtent inputExtentNames;
  NamedExtent outputExtentNames;

  memory::NodeId input;
  memory::NodeId output;

  memory::AdjGraph<ComputeTensor, ComputeOp> hypergraph;
  ModelMeta meta;

  SymGraph symGraph;
};

} // namespace denox::compiler::details::model
