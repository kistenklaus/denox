#pragma once

#include "ATen/core/ATen_fwd.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"
#include "torch/serialize/input-archive.h"
#include "torch/types.h"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include <stdexcept>

namespace vkcnn::torch {

::torch::Tensor fromActivation(ActivationHostTensorConstView activationView);
ActivationHostTensor toActivation(::torch::Tensor tensor, ActivationLayout layout = ActivationLayout::CHW);

::torch::Tensor fromFilter(FilterHostTensorConstView filterView);

FilterHostTensor toFilter(::torch::Tensor tensor);

} // namespace vkcnn::torch
