#pragma once

#include "merian/vk/context.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/ActivationShape.hpp"
#include "vkcnn/dev/utils/to_string.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include "vulkan/vulkan_handles.hpp"
#include <algorithm>
#include <compare>
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
#include <print>
#include <vector>

namespace vkcnn::dev::survey {

static constexpr std::size_t MEMORY_THROUGHPUT_CONSIDERED_FAST = 300;
static constexpr double LATENCY_CONSIDERED_FAST = 0.01;

struct ConvInOut {
  unsigned int w;
  unsigned int h;
  unsigned int c;
  unsigned int k;

  constexpr auto operator<=>(const ConvInOut &) const = default;
};

struct ConvTypes {
  FloatType inputType = FloatType::F16;
  FloatType outputType = FloatType::F16;
  FloatType filterType = FloatType::F16;

  constexpr auto operator<=>(const ConvTypes &) const = default;
};

struct ConvLayouts {
  ActivationLayout inputLayout = ActivationLayout::HWC;
  ActivationLayout outputLayout = ActivationLayout::HWC;

  constexpr auto operator<=>(const ConvLayouts &) const = default;
};

struct ConvSurveyEntry {
  bool supports = false;

  double latency = 0.0;
  double latencyStd = 0.0;

  ConvInOut inout = {};
  glm::uvec2 kernelSize = {};
  ConvTypes types = {};
  ConvLayouts layouts = {};
  std::optional<ActivationFunction> activation = std::nullopt;
};

struct ConvShaderSurvey {
  std::string name;
  std::vector<ConvSurveyEntry> entries;
};

struct ConvSurvey {
  std::vector<ConvShaderSurvey> entries;

  void print() {
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

      fmt::print("{:=^70}", fmt::format("{}x{}x{}/{}", inout.w, inout.h,
                                        inout.c, inout.k));
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
                          vkcnn::to_string(var.layouts.inputLayout), var.s,
                          var.r, vkcnn::to_string(var.types.filterType), "ReLU",
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

  void dep_print() {
    for (auto shaderSurvey : entries) {
      fmt::print("{:=^70}\n", shaderSurvey.name);
      std::map<ConvInOut, std::vector<ConvSurveyEntry>> inoutMap;
      for (auto &entry : shaderSurvey.entries) {
        inoutMap[entry.inout].push_back(entry);
      }

      for (const auto &[inout, entries] : inoutMap) {
        fmt::print("{:-^70}\n", fmt::format("{}x{}x{}/{}", inout.w, inout.h,
                                            inout.c, inout.k));
        for (const auto &entry : entries) {
          std::string configStr;
          if (entry.activation.has_value()) {
            configStr =
                fmt::format("{}:{} -{}x{}{}-> {} -> {}:{}", "F16", "CHWC8",
                            entry.kernelSize.x, entry.kernelSize.y, "F16",
                            "ReLU", "F16", "CHWC8");
          } else {
            configStr = fmt::format("{}:{} -({}x{}{})-> {}:{}", "F16", "CHWC8",
                                    entry.kernelSize.x, entry.kernelSize.y,
                                    "F16", "F16", "CHWC8");
          }
          std::print("[{:<40}]: ", configStr);
          if (!entry.supports) {
            std::println("\x1B[33mnot supported\x1B[0m");
            continue;
          }
          std::size_t memsize = entry.inout.w * entry.inout.h * entry.inout.c *
                                    entry.types.inputType.size() +
                                entry.inout.w * entry.inout.h * entry.inout.k *
                                    entry.types.outputType.size() +
                                entry.kernelSize.x * entry.kernelSize.y *
                                    entry.inout.c * entry.inout.k *
                                    entry.types.filterType.size();

          std::println(
              "\x1B[32m{:3.3f}ms  {}GB/s\x1B[0m", entry.latency,
              static_cast<std::size_t>((memsize / entry.latency) * 1e-6));
        }
      }
    }
  }
};

struct ConvMatrix {
  std::vector<ConvInOut> inout;
  std::vector<glm::uvec2> kernelSize;
  std::vector<ConvTypes> types;
  std::vector<ConvLayouts> layouts;
  std::vector<std::optional<ActivationFunction>> activationFunctions;
};

ConvSurvey conv(const merian::ContextHandle &context,
                std::span<shaders::ConvTemplate *> shaders, ConvMatrix configs);

} // namespace vkcnn::dev::survey
