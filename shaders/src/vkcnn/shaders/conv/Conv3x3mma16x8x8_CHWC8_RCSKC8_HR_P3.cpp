#include "./Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3.hpp"

#include "vkcnn/common/io/read_file.hpp"
#include <cstring>

namespace vkcnn::shaders {

Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3::Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3()
    : m_source(vkcnn::readFile("./shaders/src/vkcnn/shaders/conv/"
                               "conv3x3mma16x8x8f16_CHWC8_RCSKC8_HR_P3.comp")) {
}

bool Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3::supports(const OpConv &op) const {
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

ConvShaderSource
Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3::do_specialize(const OpConv &op) const {
  std::vector<std::byte> src{m_source.size() * sizeof(std::string::value_type)};
  std::memcpy(src.data(), m_source.data(), src.size());
  std::uint32_t specConstants[2] = {op.filterShape.c, op.filterShape.k};
  return ConvShaderSource(std::move(src), ShaderLang::GLSL,
                          SpecializationConstants{specConstants}, {},
                          glm::uvec2(16, 8), op.inputLayout, op.inputType,
                          op.outputLayout, op.outputType,
                          FilterDescriptor{
                              op.filterShape,
                              FilterLayout::RCSKC8,
                              FloatType::F16,
                          },
                          "Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3");
}
} // namespace vkcnn::shaders
