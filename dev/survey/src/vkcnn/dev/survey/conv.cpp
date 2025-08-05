#include "./conv.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/FilterDeviceTensor.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include "vkcnn/dev/utils/to_string.hpp"
#include <algorithm>
#include <optional>

using namespace vkcnn;

static constexpr std::size_t ITERATIONS = 100;

static dev::survey::ConvSurveyEntry survey_conv_template(
    const merian::ContextHandle &context, shaders::ConvTemplate *conv,
    const dev::survey::ConvInOut &inout, glm::uvec2 kernelSize,
    dev::survey::ConvTypes types, dev::survey::ConvLayouts layouts,
    dev::survey::ConvPadStride padStride,
    std::optional<ActivationFunction> activation) {
  OpConv op{FilterShape{kernelSize.x, kernelSize.y, inout.c, inout.k},
            types.filterType,
            types.biasType,
            layouts.inputLayout,
            types.inputType,
            layouts.outputLayout,
            types.outputType,
            activation,
            types.arithmeticType,
            padStride.stride,
            padStride.padding};

  auto opt = conv->specialize(op);
  if (!opt.has_value()) {
    return dev::survey::ConvSurveyEntry{false,   0.0f,       0.0f,
                                        inout,   kernelSize, types,
                                        layouts, padStride,  activation};
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
  return dev::survey::ConvSurveyEntry{true,    latency,    latencyStd,
                                      inout,   kernelSize, types,
                                      layouts, padStride,  activation};
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
            for (auto padstride : matrix.padStride) {
              for (auto activation : matrix.activationFunctions) {
                auto entry =
                    survey_conv_template(context, shader, inout, kernelSize,
                                         types, layouts, padstride, activation);
                shaderSurvey.entries.push_back(std::move(entry));
              }
            }
          }
        }
      }
    }
    survey.entries.push_back(std::move(shaderSurvey));
  }
  return survey;
}
void vkcnn::dev::survey::ConvSurvey::print() {
  struct Entry {
    double latency;
    double latencyStd;
    double throughput;
    bool isFast;
  };
  struct Variant {
    unsigned int s;
    unsigned int r;
    ConvTypes types;
    ConvLayouts layouts;
    std::optional<ActivationFunction> activation = std::nullopt;

    constexpr auto operator<=>(const Variant &) const = default;
  };
  std::map<ConvInOut, std::map<Variant, std::map<std::string_view, Entry>>>
      printMap;
  for (const auto &shaderSurvey : entries) {
    for (const auto &entry : shaderSurvey.entries) {
      Variant var{entry.kernelSize.x, entry.kernelSize.y, entry.types,
                  entry.layouts, entry.activation};
      if (entry.supports) {
        std::size_t memsize = entry.inout.w * entry.inout.h * entry.inout.c *
                                  entry.types.inputType.size() +
                              entry.inout.w * entry.inout.h * entry.inout.k *
                                  entry.types.outputType.size() +
                              entry.kernelSize.x * entry.kernelSize.y *
                                  entry.inout.c * entry.inout.k *
                                  entry.types.filterType.size();
        double throughput = (memsize / entry.latency) * 1e-6;
        bool fast = entry.latency <= LATENCY_CONSIDERED_FAST ||
                    throughput > MEMORY_THROUGHPUT_CONSIDERED_FAST;
        printMap[entry.inout][var][shaderSurvey.name] = {
            entry.latency,
            entry.latencyStd,
            throughput,
            fast,
        };
      } else {
        printMap[entry.inout][var];
      }
    }
  }
  for (const auto &[inout, varMap] : printMap) {

    bool supported = false;
    bool fastImplementation = false;
    for (const auto &[var, shaderMap] : varMap) {
      if (!shaderMap.empty()) {
        supported = true;
        for (const auto &[name, entry] : shaderMap) {
          if (entry.isFast) {
            fastImplementation = true;
          }
        }
      }
    }

    if (supported) {
      if (fastImplementation) {
        fmt::print("\x1B[32m");
      } else {
        fmt::print("\x1B[33m");
      }
    } else {
      fmt::print("\x1B[31m");
    }

    fmt::print("{:=^70}",
               fmt::format("{}x{}x{}/{}", inout.w, inout.h, inout.c, inout.k));
    fmt::print("\x1B[0m");

    fmt::println("");

    if (!supported) {
      continue;
    }
    for (const auto &[var, shaderMap] : varMap) {
      std::string configStr;
      if (var.activation.has_value()) {
        configStr =
            fmt::format("{}:{} -{}x{}{}-> {} -> {}:{}",
                        vkcnn::to_string(var.types.inputType),
                        vkcnn::to_string(var.layouts.inputLayout), var.s, var.r,
                        vkcnn::to_string(var.types.filterType), "ReLU",
                        vkcnn::to_string(var.types.outputType),
                        vkcnn::to_string(var.layouts.outputLayout));
      } else {
        configStr = fmt::format(
            "{}:{} -({}x{}{})-> {}:{}", vkcnn::to_string(var.types.inputType),
            vkcnn::to_string(var.layouts.inputLayout), var.s, var.r,
            vkcnn::to_string(var.types.filterType),
            vkcnn::to_string(var.types.outputType),
            vkcnn::to_string(var.layouts.outputLayout));
      }
      fmt::println("{:-^70}", configStr);

      if (shaderMap.empty()) {
        fmt::println("Not-supported");
      } else {
        struct Ordered {
          double latency;
          double latencyStd;
          double throughput;
          std::string_view name;

          constexpr auto operator<=>(const Ordered &) const = default;
        };

        std::vector<Ordered> ordered;
        for (const auto &[shaderName, entry] : shaderMap) {
          ordered.push_back(Ordered{entry.latency, entry.latencyStd,
                                    entry.throughput, shaderName});
        }
        std::ranges::sort(ordered);
        for (const auto &o : ordered) {
          fmt::println("[{:<50}]: {:3.3}ms {}GB/s", o.name, o.latency,
                       static_cast<std::size_t>(o.throughput));
        }
      }
    }
  }
}
