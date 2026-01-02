#pragma once

#include "denox/compiler/implement/Tensor.hpp"
namespace denox::compiler {

struct TensorView {
  uint64_t buffer;
  Sym offset;
  Sym size;
  TensorInfo info;
};

} // namespace denox::compiler
