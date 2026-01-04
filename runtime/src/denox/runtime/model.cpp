#include "model.hpp"
#include "context.hpp"
#include "dnx.h"
#include <algorithm>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

static ModelDispatch
create_model_dispatch(const runtime::ContextHandle &ctx, const dnx::Model *dnx,
                      const dnx::ComputeDispatch *dispatch) {

  std::size_t descriptorSetCount = dispatch->bindings()->size();
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts(descriptorSetCount);
  std::vector<runtime::ModelDescriptorSet> descriptorSets(descriptorSetCount);

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(8);
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    bindings.clear();
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding =
          set->bindings()->Get(static_cast<unsigned int>(b));
      VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
      descriptorSetLayoutBinding.binding = binding->binding();
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.descriptorType =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorSetLayoutBinding.pImmutableSamplers = nullptr;
      descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(descriptorSetLayoutBinding);
    }
    VkDescriptorSetLayout setLayout = ctx->createDescriptorSetLayout(bindings);
    descriptorSetLayouts[s] = setLayout;
    descriptorSets[s].descriptorSetLayout = setLayout;
    descriptorSets[s].set = set->set();
  }

  std::uint32_t pushConstantRange = dispatch->push_constant()->size();
  VkPipelineLayout pipelineLayout =
      ctx->createPipelineLayout(descriptorSetLayouts, pushConstantRange);

  std::uint32_t binaryId = dispatch->binary_id();
  const dnx::ShaderBinary *binary = dnx->shader_binaries()->Get(binaryId);
  std::span<const std::uint32_t> spirv(binary->spirv()->data(),
                                       binary->spirv()->size());
  VkPipeline pipeline = ctx->createComputePipeline(
      pipelineLayout, spirv, dispatch->entry_point()->c_str());

  return runtime::ModelDispatch{
      .dispatch = dispatch,
      .descriptorSets = std::move(descriptorSets),
      .pipelineLayout = pipelineLayout,
      .pipeline = pipeline,
  };
}

enum class TensorState {
  Undefined,
  ComputeWrite,
  ComputeRead,
};

static std::optional<runtime::ModelBarrier>
generate_pipeline_barrier([[maybe_unused]] const dnx::Model *dnx,
                          std::vector<TensorState> &tensorStates,
                          const dnx::ComputeDispatch *dispatch) {

  enum HazardType : int { None = 0, RAW = 1, WAW = 2, WAR = 4 };

  std::vector<runtime::ModelBufferBarrier> bufferBarriers;
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding =
          set->bindings()->Get(static_cast<unsigned int>(b));
      TensorState currentState = tensorStates[binding->tensor()];
      dnx::Access access = binding->access();
      HazardType hazard = None;
      TensorState nextState;
      switch (currentState) {
      case TensorState::Undefined:
        switch (access) {
        case dnx::Access_ReadOnly:
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
        case dnx::Access_ReadWrite:
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      case TensorState::ComputeWrite:
        switch (access) {
        case dnx::Access_ReadOnly:
          hazard = static_cast<HazardType>(hazard | RAW);
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        case dnx::Access_ReadWrite:
          hazard = static_cast<HazardType>(hazard | RAW);
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      case TensorState::ComputeRead:
        switch (access) {
        case dnx::Access_ReadOnly:
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
          hazard = static_cast<HazardType>(hazard | WAR);
          nextState = TensorState::ComputeWrite;
          break;
        case dnx::Access_ReadWrite:
          hazard = static_cast<HazardType>(hazard | WAR);
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      }
      if (hazard != None) {
        runtime::ModelBufferBarrier barrier;
        barrier.srcStage = VK_SHADER_STAGE_COMPUTE_BIT;
        barrier.dstStage = VK_SHADER_STAGE_COMPUTE_BIT;
        barrier.srcAccess = 0;
        barrier.dstAccess = 0;
        if (hazard & RAW) {
          barrier.srcAccess |= VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_READ_BIT;
        }
        if (hazard & WAR) {
          barrier.srcAccess |= VK_ACCESS_SHADER_READ_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_WRITE_BIT;
        }
        if (hazard & WAW) {
          barrier.srcAccess |= VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_WRITE_BIT;
        }
        barrier.tensorId = binding->tensor();
        bufferBarriers.push_back(barrier);
      }
      tensorStates[binding->tensor()] = nextState;
    }
  }
  if (bufferBarriers.empty()) {
    return std::nullopt;
  } else {
    return runtime::ModelBarrier{std::move(bufferBarriers), {}};
  }
}

static void collect_descriptor_pool_requirements(
    const dnx::ComputeDispatch *dispatch,
    ModelDescriptorPoolRequirements &requirements) {

  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *setBinding =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    for (std::size_t b = 0; b < setBinding->bindings()->size(); ++b) {
      // const dnx::DescriptorBinding *binding =
      // setBinding->bindings()->Get(static_cast<unsigned int>(b));
      const auto it = std::find_if(
          requirements.poolSizes.begin(), requirements.poolSizes.end(),
          [&](const VkDescriptorPoolSize &poolSize) {
            return poolSize.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          });
      if (it == requirements.poolSizes.end()) {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;
        requirements.poolSizes.push_back(poolSize);
      } else {
        it->descriptorCount += 1;
      }
    }
  }
  requirements.maxSets += dispatch->bindings()->size();
}

Model::Model(const ContextHandle &context, memory::span<const std::byte> dnxbuf)
    : m_context(context), m_dnxbuf(dnxbuf.begin(), dnxbuf.end()) {
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(m_dnxbuf.data()), m_dnxbuf.size());
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
    DENOX_ERROR("Failed to verify dnx file format!");
    diag::invalid_state();
  }
  m_dnx = denox::dnx::GetModel(dnxbuf.data());

  memory::vector<TensorState> tensorStates(m_dnx->tensors()->size(),
                                           TensorState::Undefined);
  for (size_t d = 0; d < m_dnx->dispatches()->size(); ++d) {
    const dnx::ComputeDispatch *dispatch =
        m_dnx->dispatches()->Get(static_cast<unsigned int>(d));
    ModelDispatch modelDispatch =
        create_model_dispatch(m_context, m_dnx, dispatch);

    if (std::optional<ModelBarrier> barrier =
            generate_pipeline_barrier(m_dnx, tensorStates, dispatch)) {
      m_cmds.emplace_back(*barrier);
    }
    m_cmds.emplace_back(modelDispatch);
    collect_descriptor_pool_requirements(dispatch,
                                         m_descriptorPoolRequirements);
  }
}

Model::~Model() { release(); }

void Model::release() {
  for (const auto &cmd : m_cmds) {
    if (std::holds_alternative<ModelDispatch>(cmd)) {
      const auto &dispatch = std::get<ModelDispatch>(cmd);
      for (const auto &set : dispatch.descriptorSets) {
        m_context->destroyDescriptorSetLayout(set.descriptorSetLayout);
      }
      m_context->destroyPipelineLayout(dispatch.pipelineLayout);
      m_context->destroyPipeline(dispatch.pipeline);
    }
  }
  m_descriptorPoolRequirements.poolSizes.clear();
  m_descriptorPoolRequirements.maxSets = 0;
  m_context = nullptr;
  m_dnxbuf.clear();
  m_dnx = nullptr;
  m_cmds.clear();
}

memory::vector<ModelOutput>
Model::infer(memory::span<const ModelInput> inputs) {
  // determine ValueSpecs. 

  return {};
}

} // namespace denox::runtime
