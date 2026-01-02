#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/memory/container/small_vector.hpp"
#include <cstdint>
namespace denox::compiler {

struct SpvDispatch {
  uint32_t binaryId;
  memory::small_vector<PushConstant, 6> pushConstants;
  Sym workgroupCountX;
  Sym workgroupCountY;
  Sym workgroupCountZ;
  memory::small_vector<TensorBinding, 4> bindings;
  ComputeDispatchInfo info;
};

} // namespace denox::compiler
