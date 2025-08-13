#pragma once

#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
namespace vkcnn {

struct OpActivation {
  ActivationLayout inputLayout;
  FloatType inputType;

  ActivationLayout outputLayout;
  FloatType outputType;

  unsigned int channels;

  ActivationFunction function;
};
} // namespace vkcnn
