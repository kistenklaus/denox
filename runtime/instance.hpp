#pragma once

#include "context.hpp"
#include "model.hpp"
#include <vector>

namespace denox::runtime {

struct InstanceDispatch {
  const ModelDispatch *dispatch;
  VkDescriptorSet *descriptorSets;
  void *pushConstantValues;
  std::uint32_t *workgroupCounts;
};

struct InstanceBarrier {
  std::vector<VkBufferMemoryBarrier> bufferBarrier;
};

using InstanceCmd = std::variant<InstanceBarrier, InstanceDispatch>;

struct InstanceTensor {
  std::uint64_t size;
  std::uint64_t offset;
  std::uint32_t buffer;
};

struct InstanceBuffer {
  runtime::Buffer buffer;
  std::uint64_t size;
};

struct InstanceTensorInfo {
  const char *name;
  std::uint32_t tensor;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

struct Instance {
  const Model *model;
  std::vector<std::int64_t> symbolValues;
  std::vector<InstanceTensorInfo> inputs;
  std::vector<InstanceTensorInfo> outputs;
  std::vector<InstanceBuffer> buffers;
  std::vector<InstanceTensor> tensors;
  VkDescriptorPool descriptorPool;
  std::vector<InstanceCmd> cmds;
};

} // namespace denox::runtime
