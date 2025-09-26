#pragma once

#include "compiler/ir/impl/ComputeDispatch.hpp"
#include "compiler/ir/impl/MemoryConstrain.hpp"
#include "compiler/ir/impl/Parameter.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "memory/container/vector.hpp"
#include "symbolic/SymGraph.hpp"

namespace denox::compiler {

struct ImplModel {
  SymGraph symGraph;

  memory::vector<TensorStorageRequirements> tensors;
  memory::vector<ComputeDispatch> dispatches;

  memory::vector<MemoryImplicitConcatConstrain> memoryImplicitConcatConstrains;
  memory::vector<Parameter> parameters;
};

} // namespace denox::compiler
