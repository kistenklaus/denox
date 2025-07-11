#include "./Conv3x3mma16x16x16f16_CHWC16_RCSKC16_HR_P1.hpp"

#include "vkcnn/common/io/read_file.hpp"
#include <cstring>

namespace vkcnn::shaders {

Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1::
    Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1()
    : m_source(
          vkcnn::readFile("./shaders/src/vkcnn/shaders/conv/"
                          "conv3x3mma16x16x16f16_CHWC16_RCSKC16_HR_P1.comp")) {}

bool Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1::supports(const OpConv &op) const {
  if (op.filterShape.r != 3)
    return false;
  if (op.filterShape.s != 3)
    return false;
  if ((op.filterShape.c % 16) != 0)
    return false;
  if ((op.filterShape.k % 16) != 0)
    return false;
  if (op.filterType != FloatType::F16)
    return false;
  if (op.inputLayout != ActivationLayout::CHWC16)
    return false;
  if (op.inputType != FloatType::F16)
    return false;

  if (op.outputLayout != ActivationLayout::CHWC16)
    return false;
  if (op.outputType != FloatType::F16)
    return false;

  if (op.activationFunc.has_value())
    return false;
  return true;
}

ConvShaderSource
Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1::do_specialize(const OpConv &op) const {
  std::vector<std::byte> src{m_source.size() * sizeof(std::string::value_type)};
  std::memcpy(src.data(), m_source.data(), src.size());
  std::uint32_t specConstants[2] = {op.filterShape.c, op.filterShape.k};
  return ConvShaderSource(std::move(src), ShaderLang::GLSL,
                          SpecializationConstants{specConstants}, {},
                          glm::uvec2(16, 8), op.inputLayout, op.inputType,
                          op.outputLayout, op.outputType,
                          FilterDescriptor{
                              op.filterShape,
                              FilterLayout::RCSKC16,
                              FloatType::F16,
                          },
                          std::string(name()));
}
} // namespace vkcnn::shaders
