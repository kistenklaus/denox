#pragma once

#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"

namespace denox::compiler::impl::details {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  unsigned int config;
  algorithm::ConstGraphMatch<TensorInstance, ComputeOp> match;
};

}
