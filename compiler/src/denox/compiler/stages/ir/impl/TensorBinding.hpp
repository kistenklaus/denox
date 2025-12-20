#pragma once

#include "compiler/ir/impl/TensorId.hpp"

namespace denox::compiler {

enum class AccessFlag {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

struct TensorBinding {
  std::uint32_t set;
  std::uint32_t binding;
  AccessFlag accessFlag;
  TensorId tensorId;
};

} // namespace denox::compiler
