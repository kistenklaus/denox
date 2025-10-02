#pragma once

#include "compiler/ir/impl/TensorId.hpp"
#include "symbolic/sym_vec2.hpp"

namespace denox::compiler {

struct OutputDesc {
  unsigned int channels;
  sym_vec2 extent;
  TensorId tensor;
};

} // namespace denox::compiler
