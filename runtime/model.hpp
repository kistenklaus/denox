#pragma once

#include <dnx.h>
#include <variant>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

struct ModelDescriptorSet {
  std::uint16_t set;
  VkDescriptorSetLayout descriptorSetLayout;
};

struct ModelDispatch {
  const dnx::ComputeDispatch *dispatch;
  const dnx::DispatchInfo* info;
  std::vector<ModelDescriptorSet> descriptorSets;
  VkPipelineLayout pipelineLayout;
  VkPipeline pipeline;
};

struct ModelBufferBarrier {
  VkPipelineStageFlags srcStage;
  VkPipelineStageFlags dstStage;
  VkAccessFlags srcAccess;
  VkAccessFlags dstAccess;
  std::uint32_t tensorId;
};

struct ModelImageMemoryBarrier {
  // TODO
};

struct ModelBarrier {
  std::vector<ModelBufferBarrier> bufferBarriers;
  std::vector<ModelImageMemoryBarrier> imageMemoryBarriers;
};

using ModelCmd = std::variant<ModelBarrier, ModelDispatch>;

struct ModelDescriptorPoolRequirements {
  std::size_t maxSets;
  std::vector<VkDescriptorPoolSize> poolSizes;
};

struct Model {
  void *dnxBuffer;
  const dnx::Model *dnx;
  std::vector<ModelCmd> cmds;
  ModelDescriptorPoolRequirements descriptorPoolRequirements;
};

} // namespace denox::runtime
