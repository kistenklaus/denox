#include "./conv.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/FilterDeviceTensor.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <algorithm>
#include <optional>

using namespace vkcnn;

static constexpr std::size_t ITERATIONS = 100;

static dev::survey::ConvSurveyEntry survey_conv_template(
    const merian::ContextHandle &context, shaders::ConvTemplate *conv,
    const dev::survey::ConvInOut &inout, glm::uvec2 kernelSize,
    dev::survey::ConvTypes types, dev::survey::ConvLayouts layouts,
    std::optional<ActivationFunction> activation) {
  OpConv op{
      FilterShape{kernelSize.x, kernelSize.y, inout.c, inout.k},
      types.filterType,
      layouts.inputLayout,
      types.inputType,
      layouts.outputLayout,
      types.outputType,
      activation,
  };

  auto opt = conv->specialize(op);
  if (!opt.has_value()) {
    return dev::survey::ConvSurveyEntry{false,      0.0f,  0.0f,    inout,
                                        kernelSize, types, layouts, activation};
  }
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

  ConvShaderSource src = *opt;
  vkcnn::runtime::ActivationDeviceTensor inputTensor{
      ActivationDescriptor{
          {inout.w, inout.h, inout.c}, layouts.inputLayout, types.inputType},
      deviceAlloc, true};
  vkcnn::runtime::ActivationDeviceTensor outputTensor{
      ActivationDescriptor{
          {inout.w, inout.h, inout.c},
          layouts.outputLayout,
          types.outputType,
      },
      deviceAlloc, true};
  vkcnn::runtime::FilterDeviceTensor filterTensor{FilterDescriptor{
                                                      op.filterShape,
                                                      src.filterLayout(),
                                                      types.filterType,
                                                  },
                                                  deviceAlloc, true};

  vkcnn::runtime::ConvPipeline convPipe{context, shaderCompiler, src,
                                        filterTensor};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  for (unsigned int i = 0; i < ITERATIONS; ++i) {
    profiler->start("conv");
    profiler->cmd_start(cmd, "conv");
    convPipe.run(cmd, inputTensor, outputTensor);
    profiler->end();
    profiler->cmd_end(cmd);
  }

  cmd->end();
  queue->submit_wait(cmd);

  profiler->collect(true, false);
  auto report = profiler->get_report().gpu_report;
  auto it =
      std::ranges::find_if(report, [](auto rep) { return rep.name == "conv"; });
  assert(it != report.end());
  auto entry = *it;

  double latency = entry.duration;
  double latencyStd = entry.std_deviation;
  return dev::survey::ConvSurveyEntry{
      true, latency, latencyStd, inout, kernelSize, types, layouts, activation};
}

vkcnn::dev::survey::ConvSurvey
vkcnn::dev::survey::conv(const merian::ContextHandle &context,
                         std::span<shaders::ConvTemplate *> shaders,
                         ConvMatrix matrix) {
  ConvSurvey survey;
  for (auto shader : shaders) {
    ConvShaderSurvey shaderSurvey;
    shaderSurvey.name = shader->name();
    for (auto inout : matrix.inout) {
      for (auto kernelSize : matrix.kernelSize) {
        for (auto types : matrix.types) {
          for (auto layouts : matrix.layouts) {
            for (auto activation : matrix.activationFunctions) {
              auto entry =
                  survey_conv_template(context, shader, inout, kernelSize,
                                       types, layouts, activation);
              shaderSurvey.entries.push_back(std::move(entry));
            }
          }
        }
      }
    }
    survey.entries.push_back(std::move(shaderSurvey));
  }
  return survey;
}
