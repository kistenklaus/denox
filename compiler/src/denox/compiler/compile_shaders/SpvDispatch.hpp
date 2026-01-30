#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/memory/container/small_vector.hpp"
#include <cstdint>
namespace denox::compiler {

struct SpvDispatch {

  static constexpr size_t PC_SVO = 6;
  static constexpr size_t BINDING_SVO = 4;

  uint32_t binaryId;
  memory::small_vector<PushConstant, PC_SVO> pushConstants;
  Sym workgroupCountX;
  Sym workgroupCountY;
  Sym workgroupCountZ;
  memory::small_vector<TensorBinding, BINDING_SVO> bindings;
  ComputeDispatchInfo info;
};

} // namespace denox::compiler
