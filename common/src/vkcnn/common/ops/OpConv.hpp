#pragma once

#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FilterShape.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <optional>
namespace vkcnn {

// NOTE: This type probably does not belong here and should be more part of
// common or something.

struct OpConv {
  FilterShape filterShape;
  FloatType filterType;

  ActivationLayout inputLayout;
  FloatType inputType;

  ActivationLayout outputLayout;
  FloatType outputType;

  std::optional<ActivationFunction> activationFunc;
};

} // namespace vkcnn::comp
