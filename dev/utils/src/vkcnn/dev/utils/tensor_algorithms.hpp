#pragma once

#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include <glm/ext/vector_uint2.hpp>
namespace vkcnn::tensor_algo {

void fill(ActivationHostTensorView tensor,
          std::function<float(unsigned int i)> func);

void fill(
    ActivationHostTensorView tensor,
    std::function<float(unsigned int w, unsigned int h, unsigned int c)> func);

inline void fill(ActivationHostTensorView tensor, float v) {
  fill(tensor, std::function([&](unsigned int) -> float { return v; }));
}

void fill(FilterHostTensorView tensor,
          std::function<float(unsigned int i)> func);

void fill(FilterHostTensorView tensor,
          std::function<float(unsigned int s, unsigned int r, unsigned int c,
                              unsigned int k)>
              func);

inline void fill(FilterHostTensorView tensor, float v) {
  fill(tensor, std::function([&](unsigned int) -> float { return v; }));
}

ActivationHostTensor conv(ActivationHostTensorConstView input,
                          FilterHostTensorConstView filter, glm::uvec2 stride = glm::uvec2(1),
                          glm::uvec2 padding = glm::uvec2(1));

void printActivation(ActivationHostTensorView tensor);

} // namespace vkcnn::tensor_algo
