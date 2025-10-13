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
  // std::vector<Buffer> buffers;
};

} // namespace denox::runtime
