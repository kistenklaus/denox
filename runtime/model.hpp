#pragma once

#include "context.hpp"
#include <cstddef>
#include <dnx.h>

namespace denox::runtime {

struct Model {
  void *dnxBuffer;
  const dnx::Model *dnx;
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  std::vector<VkPipelineLayout> pipelineLayouts;
  std::vector<VkPipeline> pipelines;
  std::vector<Buffer> initalizedBuffers; // <- some buffers might not be
};

struct ModelInstance {
  Model* model;
  std::vector<std::int64_t> vars;
  std::vector<Buffer> buffers;
  std::vector<bool> ownedBuffers;
};

} // namespace denox::runtime
