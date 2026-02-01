#pragma once

#include "denox/common/Access.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/common/ValueName.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/runtime/context.hpp"
#include "denox/symbolic/Sym.hpp"
#include "denox/symbolic/SymIR.hpp"
#include <dnx.h>
#include <memory>
#include <variant>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

struct ModelDescriptorBinding {
  uint16_t binding;
  Access access;
  uint32_t tensor;
};

struct ModelDescriptorSet {
  std::uint16_t set;
  VkDescriptorSetLayout descriptorSetLayout;
  memory::vector<ModelDescriptorBinding> bindings;
};

struct ModelDispatch {
  memory::vector<ModelDescriptorSet> descriptorSets;
  Sym workgroupCountX;
  Sym workgroupCountY;
  Sym workgroupCountZ;

  uint16_t pushConstantRange;
  memory::vector<std::pair<uint16_t, PushConstant>> pushConstants;

  VkPipelineLayout pipelineLayout;
  VkPipeline pipeline;

  memory::optional<memory::string> name;
  memory::optional<memory::string> debugInfo;
  memory::optional<memory::string> srcPath;
  memory::optional<Sym> memoryReads;
  memory::optional<Sym> memoryWrites;
  memory::optional<Sym> flops;
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
  uint32_t maxSets;
  memory::vector<VkDescriptorPoolSize> poolSizes;
};

struct ModelBuffer {
  Sym size;
  uint16_t alignment;
  memory::vector<uint32_t> tensors;
};

struct ModelTensor {
  Sym size;
  Sym offset;
  uint32_t buffer;

  memory::optional<memory::string> name;
  memory::optional<Sym> width;
  memory::optional<Sym> height;
  memory::optional<Sym> channels;
  memory::optional<TensorStorage> storage;
  memory::optional<TensorFormat> format;
  memory::optional<TensorDataType> dtype;

  bool isInput;
  bool isOutput;
  bool isParam;
};

struct ModelInitializer {
  uint32_t tensor;
  memory::vector<std::byte> data;
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

  const ContextHandle &context() const { return m_context; }
  const SymIR& symir() const { return m_symir; }
  memory::span<const ModelTensor> tensors() const { return m_tensors; }
  memory::span<const ModelBuffer> buffers() const { return m_buffers; }
  memory::span<const ModelCmd> cmds() const { return m_cmds; }
  const ModelDescriptorPoolRequirements &descriptorPoolRequirements() const {
    return m_descriptorPoolRequirements;
  }
  memory::span<const uint32_t> inputs() const { return m_inputs; }
  memory::span<const uint32_t> outputs() const { return m_outputs; }

  memory::span<const ModelInitializer> initializers() const {
    return m_initializers;
  }

  memory::span<const ValueName> valueNames() const { return m_valueNames; }

  void release();

private:
private:
  explicit Model(const ContextHandle &context,
                 memory::span<const std::byte> dnxbuf);

  ContextHandle m_context;
  // memory::vector<std::byte> m_dnxbuf;
  // const dnx::Model *m_dnx;

  SymIR m_symir;
  memory::vector<ModelTensor> m_tensors;
  memory::vector<ModelBuffer> m_buffers;
  memory::vector<uint32_t> m_inputs;
  memory::vector<uint32_t> m_outputs;

  memory::vector<ModelInitializer> m_initializers;

  memory::vector<ValueName> m_valueNames;

  memory::vector<ModelCmd> m_cmds;
  ModelDescriptorPoolRequirements m_descriptorPoolRequirements;

  std::weak_ptr<Model> m_selfhandle;
};

using ModelHandle = std::shared_ptr<Model>;

} // namespace denox::runtime
