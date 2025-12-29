#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/InputDesc.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/compiler/implement/OutputDesc.hpp"
#include "denox/compiler/implement/Parameter.hpp"
#include "denox/compiler/implement/TensorStorageRequirements.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct ImplModel {
  SymGraph symGraph;
  memory::vector<TensorStorageRequirements> tensors;
  memory::vector<ComputeDispatch> dispatches;
  memory::vector<SpirvBinary> shaderBinaries;

  memory::vector<MemoryImplicitConcatConstrain> memoryImplicitConcatConstrains; memory::vector<Parameter> parameters;

  memory::vector<InputDesc> inputs;
  memory::vector<OutputDesc> outputs;
};

} // namespace denox::compiler
