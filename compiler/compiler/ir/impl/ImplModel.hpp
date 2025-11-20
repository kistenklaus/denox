#pragma once

#include "compiler/ir/impl/ComputeDispatch.hpp"
#include "compiler/ir/impl/InputDesc.hpp"
#include "compiler/ir/impl/MemoryConstrain.hpp"
#include "compiler/ir/impl/OutputDesc.hpp"
#include "compiler/ir/impl/Parameter.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "memory/container/vector.hpp"
#include "shaders/compiler/ShaderBinary.hpp"
#include "symbolic/SymGraph.hpp"

namespace denox::compiler {

struct ImplModel {
  SymGraph symGraph;
  memory::vector<TensorStorageRequirements> tensors;
  memory::vector<ComputeDispatch> dispatches;
  memory::vector<ShaderBinary> shaderBinaries;

  memory::vector<MemoryImplicitConcatConstrain> memoryImplicitConcatConstrains;
  memory::vector<Parameter> parameters;

  memory::vector<InputDesc> inputs;
  memory::vector<OutputDesc> outputs;
};

} // namespace denox::compiler
