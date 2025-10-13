#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "model.hpp"
#include <fmt/printf.h>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

namespace denox {

int create_runtime_context(const char *deviceName, RuntimeContext *context) {
  auto *ctx = new runtime::Context(deviceName);
  *context = static_cast<void *>(ctx);
  return 0;
}

void destroy_runtime_context(RuntimeContext context) {
  auto *ctx = reinterpret_cast<runtime::Context *>(context);
  delete ctx;
}

int create_runtime_model(RuntimeContext context, const void *dnx,
                         size_t dnxSize, RuntimeModel *out_model) {
  flatbuffers::Verifier verifier(static_cast<const std::uint8_t *>(dnx),
                                 dnxSize);
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
    throw std::runtime_error("Failed to verify dnx file format.");
  }
  void *dnxBuffer = malloc(dnxSize);
  std::memcpy(dnxBuffer, dnx, dnxSize);
  const dnx::Model *dnxModel = dnx::GetModel(dnxBuffer);
  auto *model = new runtime::Model(dnxBuffer, dnxModel);

  auto ctx = reinterpret_cast<runtime::Context *>(context);

  model->pipelines.resize(model->dnx->dispatches()->size());
  for (std::size_t d = 0; d < model->dnx->dispatches()->size(); ++d) {
    std::uint8_t dispatchType = model->dnx->dispatches_type()->Get(d);
    switch (dispatchType) {
    case dnx::Dispatch_ComputeDispatch: {
      const dnx::ComputeDispatch *dispatch =
          model->dnx->dispatches()->GetAs<dnx::ComputeDispatch>(d);
      std::vector<VkDescriptorSetLayout> descriptorSetLayouts(
          dispatch->bindings()->size());
      for (std::size_t b = 0; b < dispatch->bindings()->size(); ++b) {
        const dnx::DescriptorSetBinding *setBindings =
            dispatch->bindings()->Get(b);
        std::vector<VkDescriptorSetLayoutBinding> bindings(
            setBindings->bindings()->size());
        for (std::size_t i = 0; i < setBindings->bindings()->size(); ++i) {
          const dnx::DescriptorBinding *binding =
              setBindings->bindings()->Get(i);
          bindings[i].binding = static_cast<std::uint32_t>(binding->binding());
          bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
          bindings[i].descriptorCount = 1;
        }
        descriptorSetLayouts[b] = ctx->createDescriptorSetLayout(bindings);
        model->descriptorSetLayouts.push_back(descriptorSetLayouts[b]);
      }
      std::uint32_t pushConstantSize = static_cast<std::uint32_t>(dispatch->push_constant()->size());
      VkPipelineLayout layout = ctx->createPipelineLayout(descriptorSetLayouts, pushConstantSize);
      model->pipelineLayouts.push_back(layout);

      std::span<const std::uint32_t> binary{dispatch->spirv_src()->data(),
                                            dispatch->spirv_src()->size()};
      const char *entry_point = dispatch->entry_point()->c_str();
      VkPipeline pipeline = ctx->createComputePipeline(layout, binary, entry_point);
      model->pipelines[d] = pipeline;
      break;
    }
    case dnx::Dispatch_NONE:
      break;
    default:
      throw std::runtime_error("Unreachable switch case statment.");
    }
  }

  *out_model = model;

  return 0;
}

void destroy_runtime_model(RuntimeContext context, RuntimeModel model) {
  assert(context != nullptr);
  assert(model != nullptr);
  auto ctx = reinterpret_cast<runtime::Context *>(context);
  auto m = reinterpret_cast<runtime::Model *>(model);
  for (VkDescriptorSetLayout layout : m->descriptorSetLayouts) {
    ctx->destroyDescriptorSetLayout(layout);
  }
  for (VkPipelineLayout layout : m->pipelineLayouts) {
    ctx->destroyPipelineLayout(layout);
  }
  for (VkPipeline pipeline : m->pipelines) {
    ctx->destroyPipeline(pipeline);
  }
}

int create_input_buffers(RuntimeModel model, int intputCount, Extent *extents,
                         RuntimeBuffer **inputs) {
  return -1;
}

int create_output_buffers(RuntimeModel model, RuntimeBuffer *inputs,
                          int outputCount, RuntimeBuffer **outputs) {
  return -1;
}

void destroy_buffers(RuntimeModel model, int count, RuntimeBuffer *buffers) {}

int eval(RuntimeModel model, denox::RuntimeBuffer *inputs,
         RuntimeBuffer *outputs) {
  return -1;
}

size_t get_buffer_size(RuntimeModel model, RuntimeBuffer buffer) { return -1; }

Extent get_buffer_extent(RuntimeModel model, RuntimeBuffer buffer) {
  Extent extent;
  return extent;
}

Dtype get_buffer_dtype(RuntimeModel model, RuntimeBuffer buffer) {
  Dtype dtype;
  return dtype;
}

Layout get_buffer_layout(RuntimeModel model, RuntimeBuffer buffer) {
  Layout layout;
  return layout;
}

void *map_buffer(RuntimeModel model, RuntimeBuffer buffer) { return nullptr; }

void unmap_buffer(RuntimeModel model, RuntimeBuffer buffer) {}

} // namespace denox
