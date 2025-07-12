#pragma once

#include "torch/serialize/input-archive.h"
#include "torch/types.h"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"

namespace vkcnn::torch {

::torch::Tensor fromActivation(ActivationHostTensorConstView activationView);
ActivationHostTensor
toActivation(::torch::Tensor tensor,
             ActivationLayout layout = ActivationLayout::CHW,
             FloatType type = FloatType::F16);

::torch::Tensor fromFilter(FilterHostTensorConstView filterView);

FilterHostTensor toFilter(::torch::Tensor tensor, FilterLayout layout, FloatType type);

::torch::Dtype fromType(FloatType type);

FloatType toType(::torch::Dtype dtype);

} // namespace vkcnn::torch
