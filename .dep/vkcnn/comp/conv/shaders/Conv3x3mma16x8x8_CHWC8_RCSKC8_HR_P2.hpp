#pragma once

#include "vkcnn/bad/read_file.hpp"
#include "vkcnn/comp/conv/ConvTemplate.hpp"
#include <cstring>
#include <fmt/base.h>

namespace vkcnn::comp {

class Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2 final : public ConvTemplate {
public:
  Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2()
      : m_source(bad::readFile("./src/vkcnn/comp/conv/shaders/"
                               "conv3x3mma16x8x8f16_CHWC8_RSCKC8_HR_P2.comp")) {
  }

  bool supports(const OpConv &op) const final override {
    if (op.filterShape.r != 3)
      return false;
    if (op.filterShape.s != 3)
      return false;
    if ((op.filterShape.c % 8) != 0)
      return false;
    if ((op.filterShape.k % 8) != 0)
      return false;
    if (op.filterType != FloatType::F16)
      return false;
    if (op.inputLayout != ActivationLayout::CHWC8)
      return false;
    if (op.inputType != FloatType::F16)
      return false;

    if (op.outputLayout != ActivationLayout::CHWC8)
      return false;
    if (op.outputType != FloatType::F16)
      return false;

    if (op.activationFunc.has_value())
      return false;
    return true;
  }

  ConvShaderSource do_specialize(const OpConv &op) const final override {
    std::vector<std::byte> src{m_source.size() *
                               sizeof(std::string::value_type)};
    std::memcpy(src.data(), m_source.data(), src.size());
    std::uint32_t specConstants[] = {op.filterShape.c, op.filterShape.k};
    return ConvShaderSource(std::move(src), ShaderLang::GLSL,
                            SpecializationConstants{specConstants}, {},
                            glm::uvec2(16, 8), op.inputLayout, op.inputType,
                            op.outputLayout, op.outputType,
                            FilterDescriptor{
                                op.filterShape,
                                FilterLayout::RCSKC8,
                                FloatType::F16,
                            },
                            "Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2");
  }

private:
  std::string m_source;
};

} // namespace vkcnn::comp
