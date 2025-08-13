#pragma once

#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
namespace vkcnn {

struct OpCopy {

  ActivationLayout inputLayout;
  FloatType inputType;
  unsigned int inputChannels;
  unsigned int inputChannelOffset;

  ActivationLayout outputLayout;
  FloatType outputType;
  unsigned int outputChannels;
  unsigned int outputChannelOffset;

  unsigned int channels;
};

} // namespace vkcnn
