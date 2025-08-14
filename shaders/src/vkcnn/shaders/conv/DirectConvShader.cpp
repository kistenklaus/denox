#include "./DirectConvShader.hpp"

#include "fmt/format.h"
#include "vkcnn/common/io/read_file.hpp"
#include "vkcnn/common/tensor/BiasLayout.hpp"
#include "vkcnn/shaders/preprocessing.hpp"
#include <cstring>
#include <fmt/base.h>
#include <glm/vec3.hpp>
#include <optional>
#include <stdexcept>
#include <unistd.h>

namespace vkcnn::shaders {

static std::string source() {
  std::string src =
      vkcnn::readFile("./shaders/src/vkcnn/shaders/conv/direct_conv.comp");
  return shaders::preprocess_shader_src_pragmas(src);
}

DirectConvShader::DirectConvShader(glm::uvec3 cmShape, glm::uvec3 sgTile,
                                   glm::uvec2 wgTile, bool asyncRead)
    : m_source(source()), m_cmShape(cmShape), m_sgTile(sgTile),
      m_wgTile(wgTile), m_asyncRead(asyncRead),
      m_name(fmt::format("direct_conv_WG{}x{}_SG{}x{}x{}_CM{}x{}x{}{}",
                         m_wgTile.x, m_wgTile.y, m_sgTile.x, m_sgTile.y,
                         m_sgTile.z, m_cmShape.x, m_cmShape.y, m_cmShape.z,
                         m_asyncRead ? "_async" : "")) {
  //
}

bool DirectConvShader::supports(const OpConv &_) const { return true; }

ConvShaderSource DirectConvShader::do_specialize(const OpConv &op) const {
  std::vector<std::byte> src{m_source.size() * sizeof(std::string::value_type)};
  std::memcpy(src.data(), m_source.data(), src.size());

  unsigned int inputChannels = op.filterShape.c;
  unsigned int outputChannels = op.filterShape.k;
  unsigned int kernelWidth = op.filterShape.s;
  unsigned int kernelHeight = op.filterShape.r;
  unsigned int strideX = op.stride.x;
  unsigned int strideY = op.stride.y;
  unsigned int paddingX = op.padding.x;
  unsigned int paddingY = op.padding.y;

  ActivationLayout inputLayout = op.inputLayout;
  if (inputLayout == ActivationLayout::HWC && inputChannels % 8 == 0) {
    inputLayout = ActivationLayout::HWC8; // promote layout (HWC == HWC8)
  }
  std::string inputLayoutMacro;
  std::string istype;
  unsigned int istype_size;
  if (inputLayout == ActivationLayout::HWC) {
    inputLayoutMacro = "IN_LAYOUT_HWC";
    istype = "uint16_t";
    istype_size = 2;
  } else if (inputLayout == ActivationLayout::HWC8) {
    inputLayoutMacro = "IN_LAYOUT_HWC8";
    istype = "uvec4";
    istype_size = 16;
  } else if (inputLayout == ActivationLayout::CHWC8) {
    inputLayoutMacro = "IN_LAYOUT_CHWC8";
    istype = "uvec4";
    istype_size = 16;
  } else {
    throw std::runtime_error("Unsupported inputLayout");
  }

  ActivationLayout outputLayout = op.outputLayout;
  if (outputLayout == ActivationLayout::HWC && outputChannels % 8 == 0) {
    outputLayout = ActivationLayout::HWC8;
  }
  std::string outputLayoutMacro;
  std::string ostype;
  unsigned int ostype_size;
  if (outputLayout == ActivationLayout::HWC) {
    outputLayoutMacro = "OUT_LAYOUT_HWC";
    ostype = "uint16_t";
    ostype_size = 2;
  } else if (outputLayout == ActivationLayout::HWC8) {
    outputLayoutMacro = "OUT_LAYOUT_HWC8";
    ostype = "uvec4";
    ostype_size = 16;
  } else if (outputLayout == ActivationLayout::CHWC8) {
    outputLayoutMacro = "OUT_LAYOUT_CHWC8";
    ostype = "uvec4";
    ostype_size = 16;
  } else {
    throw std::runtime_error("Unsupported inputLayout");
  }

  unsigned int subgroupSize = 32; // TODO get from VkPhysicalDevice
  unsigned int subgroupCount = m_wgTile.x * m_wgTile.y;

  std::string atype;
  if (op.arithmeticType == FloatType::F16) {
    atype = "float16_t";
  } else if (op.arithmeticType == FloatType::F32) {
    atype = "float";
  } else if (op.arithmeticType == FloatType::F64) {
    atype = "double";
  } else {
    throw std::runtime_error("Unsupported arithmetic type");
  }

  std::string activationMacro;
  if (op.activationFunc.has_value()) {
    switch (op.activationFunc.value()) {
    case ActivationFunction::ReLU:
      activationMacro = "ACTIVATION_ReLU";
      break;
    default:
      throw std::runtime_error("Activation function is not supported");
    }
  } else {
    activationMacro = "ACTIVATION_NONE";
  }

  FilterLayout filterLayout = FilterLayout::RSCK;
  std::string filterLayoutMacro;
  std::string fstype;
  unsigned int fstype_size;
  if ((inputChannels % m_cmShape.y == 0) &&
      (m_cmShape.y == 8 || m_cmShape.y == 16)) {
    if (m_cmShape.y == 8) {
      filterLayout = FilterLayout::RSCKC8;
      filterLayoutMacro = "FILTER_LAYOUT_RSCKC8";
      fstype = "uvec4";
      fstype_size = 16;
    } else if (m_cmShape.y == 16) {
      filterLayout = FilterLayout::RSCKC16;
      filterLayoutMacro = "FILTER_LAYOUT_RSCKC16";
      fstype = "uvec4";
      fstype_size = 16;
    } else {
      throw std::runtime_error("Invalid specialization");
    }
  } else if ((outputChannels % m_cmShape.z == 0) &&
             (m_cmShape.z == 8 || m_cmShape.z == 16)) {
    if (m_cmShape.z == 8) {
      filterLayout = FilterLayout::KRSCK8;
      filterLayoutMacro = "FILTER_LAYOUT_KRSCK8";
      fstype = "uvec4";
      fstype_size = 16;
    } else if (m_cmShape.z == 16) {
      filterLayout = FilterLayout::KRSCK16;
      filterLayoutMacro = "FILTER_LAYOUT_KRSCK16";
      fstype = "uvec4";
      fstype_size = 16;
    } else {
      throw std::runtime_error("Invalid specialization");
    }
  } else {
    filterLayoutMacro = "FILTER_LAYOUT_RSCK";
    fstype = "uint16_t";
    fstype_size = 2;
  }
  std::string asyncReadMacro = m_asyncRead ? "ASYNC_READ" : "NASYNC_READ";

  ShaderDefine defines[32] = {
      {"IN_CH", fmt::format("({})", inputChannels)},
      {"OUT_CH", fmt::format("({})", outputChannels)},
      {inputLayoutMacro, "(1)"},
      {outputLayoutMacro, "(1)"},
      {filterLayoutMacro, "(1)"},
      {"atype", atype},
      {"ATYPE_SIZE", fmt::format("({})", op.arithmeticType.size())},
      {"istype", istype},
      {"ISTYPE_SIZE", fmt::format("({})", istype_size)},
      {"ostype", ostype},
      {"OSTYPE_SIZE", fmt::format("({})", ostype_size)},
      {"fstype", fstype},
      {"FSTYPE_SIZE", fmt::format("({})", fstype_size)},
      {"CM_M", fmt::format("({})", m_cmShape.x)},
      {"CM_K", fmt::format("({})", m_cmShape.y)},
      {"CM_N", fmt::format("({})", m_cmShape.z)},
      {"WG_M", fmt::format("({})", m_wgTile.x)},
      {"WG_N", fmt::format("({})", m_wgTile.y)},
      {"SG_M", fmt::format("({})", m_sgTile.x)},
      {"SG_K", fmt::format("({})", m_sgTile.y)},
      {"SG_N", fmt::format("({})", m_sgTile.z)},
      {"SG_SIZE", fmt::format("({})", subgroupSize)},
      {"SG_COUNT", fmt::format("({})", subgroupCount)},
      {"KERNEL_X", fmt::format("{}", kernelWidth)},
      {"KERNEL_Y", fmt::format("{}", kernelHeight)},
      {"PADDING_X", fmt::format("{}", paddingX)},
      {"PADDING_Y", fmt::format("{}", paddingY)},
      {"STRIDE_X", fmt::format("{}", strideX)},
      {"STRIDE_Y", fmt::format("{}", strideY)},
      {activationMacro, "(1)"},
      {op.biasType.has_value() ? "USE_BIAS" : "NUSE_BIAS", "(1)"},
      {asyncReadMacro, "(1)"},
  };

  for (const auto &def : defines) {
    fmt::println("#define {} {}", def.name, def.value);
  }

  std::optional<WeightDescriptor::Bias> bias = std::nullopt;
  if (op.biasType.has_value()) {
    if (m_cmShape.z == 16) {
      // fmt::println("BiasLayout: C16");
      bias.emplace(BiasLayout::C16, *op.biasType);
    } else if (m_cmShape.z == 8) {
      // fmt::println("BiasLayout: C8");
      bias.emplace(BiasLayout::C8, *op.biasType);
    } else {
      throw std::runtime_error("Not supported");
    }
  }

  const unsigned int channelTile = m_cmShape.z * m_sgTile.z * m_wgTile.y;
  const unsigned int xtile = m_cmShape.x;
  const unsigned int ytile = m_sgTile.x * m_wgTile.x;

  std::string name(this->name());
  return ConvShaderSource(std::move(src), ShaderLang::GLSL,
                          SpecializationConstants{}, ShaderDefines{defines},
                          glm::uvec3(channelTile, xtile, ytile), op.inputLayout,
                          op.inputType, op.outputLayout, op.outputType,
                          WeightDescriptor{
                              op.filterShape,
                              filterLayout,
                              op.filterType,
                              bias,
                          },
                          op.stride, op.padding, name);
}

std::string_view DirectConvShader::name() const { return m_name; };

} // namespace vkcnn::shaders
