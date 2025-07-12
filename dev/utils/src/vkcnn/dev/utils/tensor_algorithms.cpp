#include "./tensor_algorithms.hpp"
#include "ATen/ops/conv2d.h"
#include "torch/nn/modules/conv.h"
#include "torch/torch.h"
#include "vkcnn/dev/utils/torch.hpp"
#include <fmt/base.h>

namespace vkcnn::tensor_algo {

void fill(ActivationHostTensorView tensor,
          std::function<float(unsigned int i)> func) {
  std::size_t entries = static_cast<std::size_t>(tensor.shape().c) *
                        static_cast<std::size_t>(tensor.shape().w) *
                        static_cast<std::size_t>(tensor.shape().h);
  for (std::size_t i = 0; i < entries; ++i) {
    tensor[i] = func(i);
  }
}

void fill(
    ActivationHostTensorView tensor,
    std::function<float(unsigned int w, unsigned int h, unsigned int c)> func) {
  for (unsigned int h = 0; h < tensor.shape().h; ++h) {
    for (unsigned int w = 0; w < tensor.shape().w; ++w) {
      for (unsigned int c = 0; c < tensor.shape().c; ++c) {
        tensor.at(w, h, c) = func(w, h, c);
      }
    }
  }
}

void fill(FilterHostTensorView tensor,
          std::function<float(unsigned int i)> func) {
  std::size_t entries = static_cast<std::size_t>(tensor.shape().s) *
                        static_cast<std::size_t>(tensor.shape().r) *
                        static_cast<std::size_t>(tensor.shape().c) *
                        static_cast<std::size_t>(tensor.shape().k);
  for (std::size_t i = 0; i < entries; ++i) {
    tensor[i] = func(i);
  }
}
void fill(FilterHostTensorView tensor,
          std::function<float(unsigned int s, unsigned int r, unsigned int c,
                              unsigned int k)>
              func) {
  for (unsigned int r = 0; r < tensor.shape().r; ++r) {
    for (unsigned int s = 0; s < tensor.shape().s; ++s) {
      for (unsigned int k = 0; k < tensor.shape().k; ++k) {
        for (unsigned int c = 0; c < tensor.shape().c; ++c) {
          tensor.at(s, r, c, k) = func(s, r, c, k);
        }
      }
    }
  }
}

ActivationHostTensor conv(ActivationHostTensorConstView inputView,
                          FilterHostTensorConstView filterView,
                          glm::uvec2 stride, glm::uvec2 padding) {
  auto input = vkcnn::torch::fromActivation(inputView);
  auto filter = vkcnn::torch::fromFilter(filterView);
  assert(input.is_cuda());
  assert(filter.is_cuda());

  std::int64_t strides[2] = {stride.x, stride.y};
  std::int64_t paddings[2] = {padding.x, padding.y};

  ::torch::IntArrayRef strides_ref{strides, 2};
  ::torch::IntArrayRef paddings_ref{paddings, 2};

  ::torch::Tensor output =
      ::torch::conv2d(input, filter, ::torch::Tensor(), strides_ref, paddings_ref);

  return vkcnn::torch::toActivation(output, inputView.layout());
}

void printActivation(ActivationHostTensorView tensor) {
  const auto &desc = tensor.desc();
  const auto shape = desc.shape;
  const std::size_t C = shape.c;
  const std::size_t H = shape.h;
  const std::size_t W = shape.w;

  for (std::size_t c = 0; c < C; ++c) {
    fmt::println("Channel {}", c);
    for (std::size_t h = 0; h < H; ++h) {
      for (std::size_t w = 0; w < W; ++w) {
        fmt::print("{:7.2f}", static_cast<float>(tensor.at(w, h, c)));
      }
      fmt::print("\n");
    }
    fmt::print("\n");
  }
}

} // namespace vkcnn::tensor_algo
