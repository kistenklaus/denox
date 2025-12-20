#pragma once

#include "shaders/IShader.hpp"

namespace denox::compiler::impl::details {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  unsigned int config;
  algorithm::ConstGraphMatch<TensorInstance, ComputeOp> match;
};

}
