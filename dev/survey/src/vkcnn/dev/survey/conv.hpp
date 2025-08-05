#pragma once

#include "merian/vk/context.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
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

struct ConvPadStride {
  glm::uvec2 stride = glm::uvec2(3,3);
  glm::uvec2 padding = glm::uvec2(1,1);
};

struct ConvTypes {
  FloatType inputType = FloatType::F16;
  FloatType outputType = FloatType::F16;
  FloatType filterType = FloatType::F16;
  FloatType biasType = FloatType::F16;
  FloatType arithmeticType = FloatType::F16;
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
  ConvPadStride padStride = {};
  std::optional<ActivationFunction> activation = std::nullopt;
};

struct ConvShaderSurvey {
  std::string name;
  std::vector<ConvSurveyEntry> entries;
};

struct ConvSurvey {
  std::vector<ConvShaderSurvey> entries;

  void print();
};

struct ConvMatrix {
  std::vector<ConvInOut> inout;
  std::vector<glm::uvec2> kernelSize;
  std::vector<ConvTypes> types;
  std::vector<ConvLayouts> layouts;
  std::vector<ConvPadStride> padStride;
  std::vector<std::optional<ActivationFunction>> activationFunctions;
};

ConvSurvey conv(const merian::ContextHandle &context,
                std::span<shaders::ConvTemplate *> shaders, ConvMatrix configs);

} // namespace vkcnn::dev::survey
