#pragma once

#include "denox/common/Access.hpp"
#include "denox/compiler/implement/TensorId.hpp"

namespace denox::compiler {

struct TensorBinding {
  std::uint32_t set;
  std::uint32_t binding;
  Access accessFlag;
  TensorId tensorId;
};

} // namespace denox::compiler
