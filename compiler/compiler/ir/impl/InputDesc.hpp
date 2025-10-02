#pragma once

#include "compiler/ir/impl/TensorId.hpp"
#include "symbolic/sym_vec2.hpp"
namespace denox::compiler {

struct InputDesc {
  unsigned int channels;
  sym_vec2 extent;
  TensorId tensor;
};

}
