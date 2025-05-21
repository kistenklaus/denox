#include "./memory_planning.hpp"
#include "pyvk/host/DynamicBufferRequirements.hpp"
#include <functional>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <ranges>

namespace pyvk {

void planMemory(std::span<DispatchOp> ops) {

  // NOTE: we assume that all dispatches are already topologically ordered by
  // their dependencies.

  // 1. determine the required memory per dispatch.
  // Start with the first dispatch, which defines the channels.
  //
  // Go throw all dispatches and their layers step by step
  // and accumulate a formula for memory requirements.
  // as well as the dependencies between them.

  std::vector<DynamicBufferRequirements> bufferRequirements;

  // We begin with the identity.
  std::shared_ptr<std::function<glm::uvec2(glm::uvec2)>> imageSizeFunc =
      std::make_shared<std::function<glm::uvec2(glm::uvec2)>>(
          [](const auto &wh) { return wh; });
  unsigned int channels = 0;

  for (const auto &[dispatchIdx, op] : ops | std::views::enumerate) {
    for (const auto &layer : op.layers()) {
      switch (layer->type) {
      case LayerType::None: {
        assert(false);
        break;
      }
      case LayerType::Input: {
        assert(channels == 0);
        channels = layer->info.input.channels;
        break;
      }
      case LayerType::Conv2d: {
        auto inputImageSize = imageSizeFunc;

        glm::uvec2 kernelSize = layer->info.conv2d.kernelSize;
        glm::uvec2 stride = layer->info.conv2d.stride;
        glm::uvec2 padding = layer->info.conv2d.padding;
        std::shared_ptr<std::function<glm::uvec2(glm::uvec2)>> imageSizeFunc =
            std::make_shared<std::function<glm::uvec2(glm::uvec2)>>(
                [=](const glm::uvec2 &wh) {
                  return glm::uvec2(
                      (wh.x + 2 * padding.x - kernelSize.x) / stride.x + 1,
                      (wh.y + 2 * padding.y - kernelSize.y) / stride.y + 1);
                });

        // checks if the input channels match.
        assert(channels != layer - info.conv2d.weights.shape()[1]);
        channels = layer->info.conv2d.weights.shape()[0];
        break;
      }
      case LayerType::Activation: {
        // activation has no effect on memory requirements.
        break;
      }
      case LayerType::MaxPool: {
        auto inputImageSize = imageSizeFunc;

        glm::uvec2 kernelSize = layer->info.maxPool.kernelSize;
        glm::uvec2 stride = layer->info.maxPool.stride;
        glm::uvec2 padding = layer->info.maxPool.padding;

        std::shared_ptr<std::function<glm::uvec2(glm::uvec2)>> imageSizeFunc =
            std::make_shared<std::function<glm::uvec2(glm::uvec2)>>(
                [=](const glm::uvec2 &wh) {
                  return glm::uvec2(
                      (wh.x + 2 * padding.x - kernelSize.x) / stride.x + 1,
                      (wh.y + 2 * padding.y - kernelSize.y) / stride.y + 1);
                });
        break;
      }
      case LayerType::Upsample: {
        break;
      }
      case LayerType::Concat: {
        break;
      }
      case LayerType::Output: {
        break;
      }
      }
    }

    bufferRequirements.push_back(
        DynamicBufferRequirements(channels, imageSizeFunc));
  }
}

} // namespace pyvk
