#pragma once

#include "compiler/ir/impl/ComputeDispatch.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "shaders/compiler/ShaderBinary.hpp"

namespace denox::compiler {

struct DbOp {
  memory::string shaderName;
  unsigned int pattern;
  unsigned int config;
  memory::vector<uint64_t> dispatches;
};

struct DbComputeDispatch {
  
};

struct ImplDb {
  memory::vector<TensorStorageRequirements> tensors; // TODO should not
                                                     // contain dynamic sizes.
  memory::vector<ComputeDispatch> dispatches; // TODO should not contain 
                                              // dynamic workgroup sizes or push constants.
  memory::vector<ShaderBinary> shaderBinaries;

  memory::vector<DbOp> ops; // <- may be more than one dispatch.
};

} // namespace denox::compiler
