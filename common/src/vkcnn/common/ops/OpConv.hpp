#pragma once

#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/BiasLayout.hpp"
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
  std::optional<FloatType> biasType;

  ActivationLayout inputLayout;
  FloatType inputType;

  ActivationLayout outputLayout;
  FloatType outputType;

  std::optional<ActivationFunction> activationFunc;

  FloatType arithmeticType;

  glm::uvec2 stride;
  glm::uvec2 padding;
};

} // namespace vkcnn
