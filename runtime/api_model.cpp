#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "model.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <variant>
#include <vulkan/vulkan_core.h>

namespace denox {

static runtime::ModelDispatch
create_model_dispatch(runtime::Context *ctx, const dnx::Model *dnx,
                      const dnx::ComputeDispatch *dispatch) {

  std::size_t descriptorSetCount = dispatch->bindings()->size();
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts(descriptorSetCount);
  std::vector<runtime::ModelDescriptorSet> descriptorSets(descriptorSetCount);

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(8);
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set = dispatch->bindings()->Get(s);
    bindings.clear();
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding = set->bindings()->Get(b);
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
  fmt::println("size = {}", dispatch->push_constant()->size());
  fmt::println("fields = {}", dispatch->push_constant()->fields()->size());
  VkPipelineLayout pipelineLayout =
      ctx->createPipelineLayout(descriptorSetLayouts, pushConstantRange);

  std::uint32_t binaryId = dispatch->binary_id();
  const dnx::ShaderBinary *binary = dnx->shader_binaries()->Get(binaryId);
  fmt::println("binaryId = {}", binaryId);
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
generate_pipeline_barrier(const dnx::Model *dnx,
                          std::vector<TensorState> &tensorStates,
                          const dnx::ComputeDispatch *dispatch) {

  enum HazardType : int { None = 0, RAW = 1, WAW = 2, WAR = 4 };

  std::vector<runtime::ModelBufferBarrier> bufferBarriers;
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set = dispatch->bindings()->Get(s);
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding = set->bindings()->Get(b);
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

static void
collect_descriptor_pool_requirements(const dnx::ComputeDispatch *dispatch,
                                     runtime::Model *m) {
  auto &requirements = m->descriptorPoolRequirements;

  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *setBinding = dispatch->bindings()->Get(s);
    for (std::size_t b = 0; b < setBinding->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding = setBinding->bindings()->Get(b);
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

int create_runtime_model(RuntimeContext context, const void *dnx_buf,
                         size_t dnxSize, RuntimeModel *out_model) {
  flatbuffers::Verifier verifier(static_cast<const std::uint8_t *>(dnx_buf),
                                 dnxSize);
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
#ifndef NDEBUG
    fmt::println("Failed to verify dnx file format.");
#endif
    return -1;
  }
  void *dnxBuffer = malloc(dnxSize);
  std::memcpy(dnxBuffer, dnx_buf, dnxSize);
  const dnx::Model *dnxModel = dnx::GetModel(dnxBuffer);
  auto *m = new runtime::Model(dnxBuffer, dnxModel);
  auto ctx = reinterpret_cast<runtime::Context *>(context);

  const auto *dnx = m->dnx;

  std::vector<TensorState> tensorStates(dnx->tensors()->size(),
                                        TensorState::Undefined);

  for (std::size_t d = 0; d < dnx->dispatches()->size(); ++d) {
    dnx::Dispatch dispatch_type =
        dnx->dispatches_type()->GetEnum<dnx::Dispatch>(d);
    switch (dispatch_type) {
    case dnx::Dispatch_ComputeDispatch: {
      const dnx::ComputeDispatch *dispatch =
          dnx->dispatches()->GetAs<dnx::ComputeDispatch>(d);

      // Create pipeline and layouts.
      runtime::ModelDispatch modelDispatch =
          create_model_dispatch(ctx, dnx, dispatch);

      // Schedule pipeline barrier cmd.
      if (std::optional<runtime::ModelBarrier> barrier =
              generate_pipeline_barrier(dnx, tensorStates, dispatch)) {
        m->cmds.push_back(*barrier);
      }

      // Schedule dispatch cmd.
      m->cmds.push_back(modelDispatch);

      collect_descriptor_pool_requirements(dispatch, m);
      break;
    }
    case dnx::Dispatch_NONE:
    default:
      throw std::runtime_error("invalid dnx file format!");
    }
  }

  *out_model = m;
  return 0;
}

void destroy_runtime_model(RuntimeContext context, RuntimeModel model) {
  assert(context != nullptr);
  assert(model != nullptr);
  auto ctx = reinterpret_cast<runtime::Context *>(context);
  auto m = reinterpret_cast<runtime::Model *>(model);

  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  std::vector<VkPipelineLayout> pipelineLayouts;
  std::vector<VkPipeline> pipelines;
  for (const auto &cmd : m->cmds) {
    if (std::holds_alternative<runtime::ModelDispatch>(cmd)) {
      const runtime::ModelDispatch &dispatch =
          std::get<runtime::ModelDispatch>(cmd);
      for (const auto &set : dispatch.descriptorSets) {
        descriptorSetLayouts.push_back(set.descriptorSetLayout);
      }
      pipelineLayouts.push_back(dispatch.pipelineLayout);
      pipelines.push_back(dispatch.pipeline);
    }
  }

  for (const auto &pipeline : pipelines) {
    ctx->destroyPipeline(pipeline);
  }

  for (const auto &pipelineLayout : pipelineLayouts) {
    ctx->destroyPipelineLayout(pipelineLayout);
  }

  for (const auto &setLayout : descriptorSetLayouts) {
    ctx->destroyDescriptorSetLayout(setLayout);
  }

  free(m->dnxBuffer);
  delete m;
}

} // namespace denox
