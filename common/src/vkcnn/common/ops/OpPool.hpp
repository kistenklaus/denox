#pragma once

#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <glm/vec2.hpp>
namespace vkcnn {

struct OpPool {
  ActivationLayout inputLayout;
  FloatType inputType;

  ActivationLayout outputLayout;
  FloatType outputType;

  unsigned int channels;

  glm::uvec2 kernelSize;
  glm::uvec2 stride;
  glm::uvec2 padding;

  PoolFunction poolingFunc;
};

} // namespace vkcnn
