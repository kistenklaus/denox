

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"

#include "ExtensionCoopMat.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/ActivationShape.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include "vkcnn/comp/conv/OpConv.hpp"
#include "vkcnn/comp/conv/shaders/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include <memory>

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
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  unsigned int W = 480;
  unsigned int H = 540;
  unsigned int C = 160;
  unsigned int K = 96;

  unsigned int R = 3;
  unsigned int S = 3;

  vkcnn::ActivationHostTensor inputHost{{vkcnn::ActivationShape{W, H, C},
                                         vkcnn::ActivationLayout::CHWC8,
                                         vkcnn::FloatType::F16}};

  vkcnn::ActivationHostTensor outputHost{{vkcnn::ActivationShape{W, H, K},
                                          vkcnn::ActivationLayout::CHWC8,
                                          vkcnn::FloatType::F16}};

  vkcnn::comp::Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2 conv;
  vkcnn::ConvShaderSource convSrc =
      conv.specialize(vkcnn::comp::OpConv{{S, R, C, K},
                                          vkcnn::FloatType::F16,
                                          inputHost.layout(),
                                          inputHost.type(),
                                          outputHost.layout(),
                                          outputHost.type(),
                                          std::nullopt})
          .value();

  vkcnn::FilterHostTensor filterHost{convSrc.filterDesc()};

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::FilterDeviceTensor filterDevice(filterHost.desc(),
                                                  deviceAlloc);

  vkcnn::runtime::ConvPipeline convPipe{context, shaderCompiler, convSrc,
                                        filterDevice};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  filterDevice.upload(cmd, filterHost);
  inputDevice.upload(cmd, inputHost);

  convPipe.run(cmd, inputDevice, outputDevice);

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  return 0;
}
