#include "./Conv3x3mmaVectorized.hpp"

#include "vkcnn/common/io/read_file.hpp"
#include <cstring>

namespace vkcnn::shaders {

Conv3x3mmaVectorized::Conv3x3mmaVectorized(glm::uvec3 mmaShape)
    : m_source(vkcnn::readFile("./shaders/src/vkcnn/shaders/conv/"
                               "conv3x3mmaVectorized.comp")),
      m_mmaShape(mmaShape) {
  assert(mmaShape.x == 16); // Other shapes are not implemented yet.
}

bool Conv3x3mmaVectorized::supports(const OpConv &op) const {
  if (op.filterShape.r != 3)
    return false;
  if (op.filterShape.s != 3)
    return false;
  if ((op.filterShape.c % 8) != 0)
    return false;

  if (op.filterShape.c > 128)
    return false; // numerically instable

  if ((op.filterShape.k % 8) != 0)
    return false;
  if (op.filterType != FloatType::F16)
    return false;
  if (op.inputLayout != ActivationLayout::CHWC8 &&
      op.inputLayout != ActivationLayout::CHWC16)
    return false;

  if (op.inputType != FloatType::F16)
    return false;

  if (op.outputLayout != ActivationLayout::CHWC8 &&
      op.outputLayout != ActivationLayout::CHWC16)
    return false;

  if (op.outputType != FloatType::F16)
    return false;

  if (op.inputLayout != op.outputLayout)
    return false;

  if (op.activationFunc.has_value())
    return false;


  if (m_mmaShape.y == 16 && op.inputLayout != ActivationLayout::CHWC16) {
    return false;
  }
  if (m_mmaShape.y == 8 && op.inputLayout != ActivationLayout::CHWC8) {
    return false;
  }

  if (m_mmaShape.z == 16 && op.outputLayout != ActivationLayout::CHWC16) {
    return false;
  }
  if (m_mmaShape.z == 8 && op.outputLayout != ActivationLayout::CHWC8) {
    return false;
  }

  return true;
}

ConvShaderSource Conv3x3mmaVectorized::do_specialize(const OpConv &op) const {
  std::vector<std::byte> src{m_source.size() * sizeof(std::string::value_type)};
  std::memcpy(src.data(), m_source.data(), src.size());
  std::uint32_t specConstants[5] = {op.filterShape.c, op.filterShape.k,
                                    m_mmaShape.x, m_mmaShape.y, m_mmaShape.z};
  return ConvShaderSource(std::move(src), ShaderLang::GLSL,
                          SpecializationConstants{specConstants}, {},
                          glm::uvec2(16, 8), op.inputLayout, op.inputType,
                          op.outputLayout, op.outputType,
                          FilterDescriptor{
                              op.filterShape,
                              FilterLayout::RCSKC8,
                              FloatType::F16,
                          },
                          "Conv3x3mmaVectorized");
}
} // namespace vkcnn::shaders
