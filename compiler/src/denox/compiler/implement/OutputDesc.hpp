#pragma once

#include "denox/compiler/implement/TensorId.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include "denox/symbolic/sym_vec2.hpp"

namespace denox::compiler {

struct OutputDesc {
  unsigned int channels;
  sym_vec2 extent;
  TensorId tensor;
  memory::ActivationLayout layout;
  memory::Dtype dtype;
};

} // namespace denox::compiler
