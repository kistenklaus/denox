#pragma once

#include "vkcnn/ImageTensor.hpp"
#include "vkcnn/WeightTensor.hpp"
namespace reference {

void conv(const vkcnn::ImageTensor& input, vkcnn::ImageTensor& output,
          const vkcnn::WeightTensor& weights, glm::uvec2 kernelSize,
          glm::uvec2 padding);
}
