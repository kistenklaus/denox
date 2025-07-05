

#include "ExtensionCoopMat.hpp"
#include "conv.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vkcnn/ImageTensor.hpp"
#include "vkcnn/WeightTensor.hpp"
#include "vkcnn/device/OpComputePipe.hpp"
#include "vkcnn/device/conv2d/Conv2d.hpp"
#include "vkcnn/device/tensor/ImageTensor.hpp"
#include "vkcnn/host/DynamicImageTensor.hpp"
#include "vkcnn/host/ImageTensorLayout.hpp"
#include "vkcnn/host/ShaderSource.hpp"
#include "vkcnn/host/WeightTensorLayout.hpp"
#include "vkcnn/host/codegen/conv2d/direct.hpp"
#include "vkcnn/host/conv2d/Conv2dSource.hpp"
#include "vkcnn/host/fprec.hpp"
#include "vkcnn/host/ops/OpConv2d.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <print>
#include <random>

/// Just a helper function, which does the initalization
merian::ContextHandle createContext() {
  // Setup logging
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
  spdlog::set_level(spdlog::level::debug);
#endif

  // Setup Vulkan context
  const auto core =
      std::make_shared<merian::ExtensionVkCore>(std::set<std::string>{
          "vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope",
          "vk12/shaderBufferInt64Atomics", "vk12/shaderSubgroupExtendedTypes"});

  const auto floatAtomics =
      std::make_shared<merian::ExtensionVkFloatAtomics>(std::set<std::string>{
          "shaderBufferFloat32Atomics",
          "shaderBufferFloat32AtomicAdd",
      });

  const auto coopMat = std::make_shared<ExtensionCoopMat>();

  const auto debug_utils =
      std::make_shared<merian::ExtensionVkDebugUtils>(true);
  const auto resources = std::make_shared<merian::ExtensionResources>();
  const auto push_descriptor =
      std::make_shared<merian::ExtensionVkPushDescriptor>();

  const std::vector<std::shared_ptr<merian::Extension>> extensions = {
      core, floatAtomics, resources, debug_utils, push_descriptor, coopMat};

  const merian::ContextHandle context = merian::Context::create(
      extensions, "vkcnn-sandbox", VK_MAKE_VERSION(1, 0, 0), 1,
      VK_API_VERSION_1_3, false);

  if (!context) {
    throw std::runtime_error("Failed to create context!!!");
  }
  return context;
}

int main() {

  // Setup
  merian::ContextHandle context = createContext();
  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle alloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  // Sandbox
  std::mt19937 prng;
  std::uniform_real_distribution<float> dist{0.0f, 1.0f};

  using namespace vkcnn;

  const unsigned int TILE_W = 16;
  const unsigned int TILE_H = 8;

  const unsigned int W = 1920;
  const unsigned int H = 1080;
  // const unsigned int W = 32;
  // const unsigned int H = 16;
  const unsigned int C = 32;
  const unsigned int K = 32;
  const unsigned int R = 3;
  const unsigned int S = 3;

  std::string shaderPath = "sandbox/conv3x3mma16x8x8f16_CHWC8_RSCKC8.comp";

  WeightTensorLayout weightLayout = WeightTensorLayout::RSCKC8;
  WeightTensor weightTensor{weightLayout, FPrec::F16, K, C, S, R};
  weightTensor.fill<float>([&](auto) { return dist(prng); });

  ImageTensorLayout inputLayout = ImageTensorLayout::CHWC8;
  ImageTensor inputTensor{inputLayout, FPrec::F16, W, H, C};
  inputTensor.fill<float>([&](auto) { return dist(prng); });

  ImageTensorLayout outputLayout = ImageTensorLayout::CHWC8;
  ImageTensor outputTensor{outputLayout, FPrec::F16, W, H, K};

  merian::PipelineLayoutBuilder pipelineLayoutBuilder{context};

  // input / output storage buffer descriptor set.
  merian::DescriptorSetLayoutBuilder set0Builder;
  set0Builder.add_binding_storage_buffer(); // input tensor.
  set0Builder.add_binding_storage_buffer(); // output tensor.
  set0Builder.add_binding_storage_buffer(); // weight tensor.

  pipelineLayoutBuilder.add_descriptor_set_layout(
      set0Builder.build_push_descriptor_layout(context));

  pipelineLayoutBuilder.add_push_constant<glm::uvec2>();

  merian::PipelineLayoutHandle pipelineLayout =
      pipelineLayoutBuilder.build_pipeline_layout();

  merian::ShaderModuleHandle shaderModule =
      shaderCompiler->find_compile_glsl_to_shadermodule(context, shaderPath);

  auto pipeline =
      std::make_shared<merian::ComputePipeline>(pipelineLayout, shaderModule);

  inputTensor.enableDevice(alloc);
  outputTensor.enableDevice(alloc);
  weightTensor.enableDevice(alloc);

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  weightTensor.flushToDevice(cmd);
  inputTensor.flushToDevice(cmd);

  cmd->bind(pipeline);
  cmd->push_constant<glm::uvec2>(pipeline, glm::uvec2(W, H));
  cmd->push_descriptor_set(pipeline, inputTensor.getDeviceHandle(),
                           outputTensor.getDeviceHandle(),
                           weightTensor.getDeviceHandle());

  for (std::size_t i = 0; i < 1; ++i) {
    profiler->start("conv3x3");
    profiler->cmd_start(cmd, "conv3x3");

    const glm::uvec2 c =
        (glm::uvec2(W, H) + glm::uvec2(TILE_W, TILE_H) - glm::uvec2(1)) /
        glm::uvec2(TILE_W, TILE_H);
    cmd->dispatch(c.x * c.y);

    profiler->end();
    profiler->cmd_end(cmd);
  }

  outputTensor.flushToHost(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  profiler->collect(true);
  auto reportStr = merian::Profiler::get_report_str(profiler->get_report());

  outputTensor.disableDevice();

  if (W <= 32) {
    ImageTensor outputTensorReference{outputLayout, FPrec::F16, W, H, K};
    reference::conv(inputTensor, outputTensorReference, weightTensor,
                    glm::uvec2(S, R), glm::uvec2(1, 1));

    fmt::println("Output tensor reference :\n{}", outputTensorReference);

    fmt::println("Output tensor:\n{}", outputTensor);

    auto diff = outputTensorReference - outputTensor;
    fmt::println("Diff: \n{}", diff);
  }
  fmt::println("{}", reportStr);

  auto optimalLat = (inputTensor.byteSize() + outputTensor.byteSize()) / 504e6;
  fmt::println("Optimal memory latency: {}ms", optimalLat);

  auto goodLat = (inputTensor.byteSize() + outputTensor.byteSize()) / 400e6;
  fmt::println("Optimization Goal: {}ms", goodLat);

  return 0;
}
