#pragma once

#include "denox/common/ValueSpec.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"
#include "denox/runtime/context.hpp"
#include <dnx.h>
#include <memory>
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

struct ModelInput {
  std::string name;
  memory::ActivationTensorConstView tensor;
};

struct ModelOutput {
  std::string name;
  memory::ActivationTensor tensor;
};

class Model {
public:
  static std::shared_ptr<Model> make(memory::span<const std::byte> dnxbuf) {
    return make(dnxbuf, Context::make(nullptr, ApiVersion::VULKAN_1_4));
  }

  static std::shared_ptr<Model> make(memory::span<const std::byte> dnxbuf,
                                     ContextHandle context) {
    auto sptr = std::shared_ptr<Model>(new Model(context, dnxbuf));
    sptr->m_selfhandle = sptr;
    return sptr;
  }

  ~Model();
  Model(const Model &o) = delete;
  Model &operator=(const Model &o) = delete;
  Model(Model &&o) = delete;
  Model &operator=(Model &&o) = delete;

  memory::vector<ModelOutput> infer(std::initializer_list<ModelInput> inputs) {
    return infer(memory::span{inputs});
  }
  memory::vector<ModelOutput> infer(memory::span<const ModelInput> inputs);

  void release();

private:
private:
  explicit Model(const ContextHandle &context,
                 memory::span<const std::byte> dnxbuf);

  ContextHandle m_context;
  memory::vector<std::byte> m_dnxbuf;
  const dnx::Model *m_dnx;
  std::vector<ModelCmd> m_cmds;
  ModelDescriptorPoolRequirements m_descriptorPoolRequirements;
  std::weak_ptr<Model> m_selfhandle;
};

using ModelHandle = std::shared_ptr<Model>;

} // namespace denox::runtime
