#pragma once

#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
#include "denox/compiler/frontend/model/DynamicInputExtent.hpp"
#include "denox/compiler/frontend/model/ModelMeta.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler::details::model {

struct ModelControlBlock {
  static constexpr memory::NodeId NullNode{static_cast<std::size_t>(-1)};
  ModelControlBlock()
      : input(NullNode), output(NullNode), hypergraph(), symGraph() {}

  NamedExtent inputExtentNames;
  NamedExtent outputExtentNames;

  std::string inputName;
  std::string outputName;

  memory::NodeId input;
  memory::NodeId output;

  memory::AdjGraph<ComputeTensor, ComputeOp> hypergraph;
  ModelMeta meta;

  SymGraph symGraph;
};

} // namespace denox::compiler::details::model
