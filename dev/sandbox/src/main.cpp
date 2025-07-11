
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "torch/cuda.h"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include "vkcnn/dev/utils/merian.hpp"
#include "vkcnn/dev/utils/tensor_algorithms.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/FilterDeviceTensor.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x16x16f16_CHWC16_RCSKC16_HR_P1.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2.hpp"

int main() {

  // Setup
  merian::ContextHandle context = vkcnn::merian::createContext();

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

  unsigned int W = 1920;
  unsigned int H = 1080;
  unsigned int C = 128;
  unsigned int K = 128;

  unsigned int R = 3;
  unsigned int S = 3;

  vkcnn::ActivationHostTensor inputHost{{vkcnn::ActivationShape{W, H, C},
                                         vkcnn::ActivationLayout::CHWC16,
                                         vkcnn::FloatType::F16}};

  vkcnn::ActivationHostTensorView view{inputHost};

  vkcnn::tensor_algo::fill(inputHost, 1.0f);

  vkcnn::ActivationHostTensor outputHost{{vkcnn::ActivationShape{W, H, K},
                                          vkcnn::ActivationLayout::CHWC16,
                                          vkcnn::FloatType::F16}};

  vkcnn::shaders::Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1 conv;
  vkcnn::ConvShaderSource convSrc =
      conv.specialize(vkcnn::OpConv{{S, R, C, K},
                                    vkcnn::FloatType::F16,
                                    inputHost.layout(),
                                    inputHost.type(),
                                    outputHost.layout(),
                                    outputHost.type(),
                                    std::nullopt})
          .value();

  vkcnn::FilterHostTensor filterHost{convSrc.filterDesc()};
  vkcnn::tensor_algo::fill(filterHost, 1.0f);

  auto output = vkcnn::tensor_algo::conv(inputHost, filterHost, glm::uvec2(1),
                                         glm::uvec2(1));
  // vkcnn::tensor_algo::printActivation(output);

  return 0;

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

  for (unsigned int i = 0; i < 10; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "conv-pipe");
    convPipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  vkcnn::tensor_algo::printActivation(outputHost);

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));

  // vkcnn::tensor_algo::printActivation(outputHost);

  return 0;
}
