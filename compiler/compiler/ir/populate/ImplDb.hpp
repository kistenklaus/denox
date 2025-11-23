#pragma once

#include "compiler/ir/impl/ComputeDispatch.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "shaders/compiler/ShaderBinary.hpp"

namespace denox::compiler {

struct DbOp {
  memory::string shaderName;
  unsigned int pattern;
  unsigned int config;
  memory::vector<uint32_t> dispatches;
};

struct DbTensorBinding {
  uint32_t set;
  uint32_t binding;
  AccessFlag access;

  uint64_t tensorByteSize;
  unsigned int tensorMinAlignment;
};

struct DbComputeDispatch {
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
  uint32_t binaryId;
  memory::vector<DbTensorBinding> bindings;
  memory::vector<std::uint8_t> pushConstant;
};

struct ImplDb {
  memory::vector<ShaderBinary> shaderBinaries;  // <- what is executed.
  memory::vector<DbComputeDispatch> dispatches; // <- what is benchmarked.
  memory::vector<DbOp> ops;                     // <- what is queried.
};

} // namespace denox::compiler
