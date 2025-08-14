#pragma once

#include "vkcnn/common/FilterMode.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
namespace vkcnn {

struct OpUpsample {
  ActivationLayout inputLayout;
  FloatType inputType;

  ActivationLayout outputLayout;
  FloatType outputType;

  unsigned int channels;
  unsigned int scalingFactor;

  FilterMode filterMode;
};

} // namespace vkcnn
