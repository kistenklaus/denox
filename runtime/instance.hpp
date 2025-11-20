#pragma once

#include "context.hpp"
#include "denox/common/types.hpp"
#include "model.hpp"
#include <vector>

namespace denox::runtime {

struct InstanceDispatch {
  const ModelDispatch *dispatch;
  VkDescriptorSet *descriptorSets;
  void *pushConstantValues;
  std::uint32_t *workgroupCounts;
  std::optional<std::string> debug_info;
  std::optional<uint64_t> memory_reads;
  std::optional<uint64_t> memory_writes;
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
  Extent width;
  Extent height;
  Extent channels;
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
