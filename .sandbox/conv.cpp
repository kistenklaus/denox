#include "./conv.hpp"
#include "vkcnn/WeightTensor.hpp"
#include <fmt/base.h>

static float getInput(const vkcnn::ImageTensor& tensor, glm::uvec2 pixel,
                      std::size_t c) {
  if (pixel.x >= tensor.w() || pixel.y >= tensor.h()) {
    return 0.0f;
  }
  assert(c < tensor.c());
  return tensor.get<float>(pixel.x, pixel.y, c);
}

static float getWeight(const vkcnn::WeightTensor& tensor, glm::uvec2 sr, std::size_t k,
                       std::size_t c) {
  return tensor.get<float>(k, c, sr.x, sr.y);
}

static void setOutput(vkcnn::ImageTensor& tensor, glm::uvec2 pixel,
                      std::size_t c, float v) {
  tensor.set<float>(pixel.x, pixel.y, c, v);
}

void reference::conv(const vkcnn::ImageTensor& input, vkcnn::ImageTensor& output,
                     const vkcnn::WeightTensor& weights, glm::uvec2 kernelSize,
                     glm::uvec2 padding) {
  const uint W = input.w();
  const uint H = input.h();
  const uint C = input.c();
  const uint K = output.c(); // output channels == K

  for (int y = 0; y < static_cast<int>(H); ++y) {
    for (int x = 0; x < static_cast<int>(W); ++x) {
      auto pixel = glm::uvec2(x, y);
      for (uint k = 0; k < K; ++k) {
        float acc = 0.0f;
        for (uint r = 0; r < kernelSize.y; ++r) {
          for (uint s = 0; s < kernelSize.x; ++s) {
            auto sr = glm::uvec2(s, r);
            auto pos = pixel + sr - padding;
  
            for (uint c = 0; c < C; ++c) {
              float v = getInput(input, pos, c);
              float w = getWeight(weights, sr, k, c);
              acc += v * w;
            }
          }
        }
  
        setOutput(output, glm::uvec2(x, y), k, acc);
      }
    }
  }
}
