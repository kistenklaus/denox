#pragma once

#include "denox/common/TensorDescriptor.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ModelInterfaceDescriptor.hpp"
#include "denox/compiler/frontend/model/ModelMeta.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler::details::model {

struct ModelControlBlock {

  std::vector<ModelInterfaceDescriptor> inputs;
  std::vector<ModelInterfaceDescriptor> outputs;

  std::vector<NamedValue> valueNames;

  memory::AdjGraph<TensorDescriptor, ComputeOp> hypergraph;

  ModelMeta meta;
  SymGraph symGraph;
};

} // namespace denox::compiler::details::model
