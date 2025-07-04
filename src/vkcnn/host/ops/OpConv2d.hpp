#pragma once

#include "vkcnn/host/DynamicWeightTensor.hpp"
#include "vkcnn/host/ImageTensorLayout.hpp"
#include "vkcnn/host/ops/TensorId.hpp"
#include "vkcnn/host/ops/PaddingMode.hpp"
#include <glm/vec2.hpp>
namespace vkcnn {

// Contains everything required to generate the code.
struct OpConv2d {
  enum class Impl {
    DIRECT,
  };

  host::DynamicWeightTensor weights;
  glm::uvec2 stride;
  glm::uvec2 padding;
  PaddingMode paddingMode;

  FPrec inputPrecision;
  ImageTensorLayout inputLayout;

  ImageTensorLayout outputLayout;
  FPrec outputPrecision;

  glm::uvec2 tileSize;

  TensorId input;
  TensorId output;
};

} // namespace vkcnn
