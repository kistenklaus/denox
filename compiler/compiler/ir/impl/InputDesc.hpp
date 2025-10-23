#pragma once

#include "compiler/ir/impl/TensorId.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "symbolic/sym_vec2.hpp"
namespace denox::compiler {

struct InputDesc {
  unsigned int channels;
  sym_vec2 extent;
  TensorId tensor;
  memory::ActivationLayout layout;
  memory::Dtype dtype;
};

}
