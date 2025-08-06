#include "./ConvGEMM.hpp"

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
      vkcnn::readFile("./shaders/src/vkcnn/shaders/conv/conv_gemm.comp");
  return shaders::preprocess_shader_src_pragmas(src);
}

ConvGEMM::ConvGEMM(glm::uvec3 cmShape, glm::uvec3 sgTile, glm::uvec2 wgTile,
                   bool asyncRead)
    : m_source(source()), m_cmShape(cmShape), m_sgTile(sgTile),
      m_wgTile(wgTile), m_asyncRead(asyncRead) {
  //
}

bool ConvGEMM::supports(const OpConv &_) const { return true; }

ConvShaderSource ConvGEMM::do_specialize(const OpConv &op) const {
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

  auto actiLayoutOrd = [](ActivationLayout layout) {
    if (layout == ActivationLayout::HWC) {
      return 0;
    } else if (layout == ActivationLayout::CHW) {
      throw std::runtime_error("Not supported");
    } else if (layout == ActivationLayout::CHWC4) {
      throw std::runtime_error("Not supported");
    } else if (layout == ActivationLayout::CHWC8) {
      return 3;
    } else if (layout == ActivationLayout::CHWC16) {
      throw std::runtime_error("Not supported");
    } else {
      throw std::runtime_error("Unsupported specialization");
    }
  };
  unsigned int inLayout = actiLayoutOrd(op.inputLayout);
  unsigned int outLayout = actiLayoutOrd(op.outputLayout);

  unsigned int subgroupSize = 32; // TODO get from VkPhysicalDevice
  unsigned int subgroupCount = m_wgTile.x * m_wgTile.y;

  std::uint32_t specConstants[8] = {
      inputChannels,  //
      outputChannels, //
      kernelWidth,    //
      kernelHeight,   //
      strideX,        //
      strideY,        //
      paddingX,       //
      paddingY,       //
  };
  fmt::println("Padding = ({},{})", op.padding.x, op.padding.y);
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

  auto deriveSType =
      [](unsigned int channelSize) -> std::pair<std::string, unsigned int> {
    if (channelSize % 16 == 0) {
      return {"uvec4", 16};
    } else if (channelSize % 2 == 0) {
      return {"uint16_t", 2};
    } else {
      throw std::runtime_error("Unsupported channelSize");
    }
  };
  auto [istype, istypesize] = deriveSType(inputChannels * op.inputType.size());
  auto [ostype, ostypesize] =
      deriveSType(outputChannels * op.outputType.size());

  unsigned int actiFuncOrd;
  if (op.activationFunc.has_value()) {
    switch (op.activationFunc.value()) {
    case ActivationFunction::ReLU:
      actiFuncOrd = 1;
      break;
    default:
      throw std::runtime_error("Activation function is not supported");
    }
  } else {
    actiFuncOrd = 0;
  }

  // assert CM_M8 * CM_K * SG_M * SG_K % WG_N == 0
  if (op.inputLayout == ActivationLayout::HWC &&
      (m_cmShape.x / 8 * m_cmShape.y * m_sgTile.x * m_sgTile.y) % m_wgTile.y !=
          0) {
    throw std::runtime_error("Not supported");
  }

  ShaderDefine defines[21] = {
      {"IN_LAYOUT", fmt::format("({})", inLayout)},
      {"OUT_LAYOUT", fmt::format("({})", outLayout)},
      {"atype", atype},
      {"ATYPE_SIZE", fmt::format("({})", op.arithmeticType.size())},
      {"istype", istype},
      {"ISTYPE_SIZE", fmt::format("({})", istypesize)},
      {"ostype", ostype},
      {"OSTYPE_SIZE", fmt::format("({})", ostypesize)},
      {"CM_M", fmt::format("({})", m_cmShape.x)},
      {"CM_K", fmt::format("({})", m_cmShape.y)},
      {"CM_N", fmt::format("({})", m_cmShape.z)},
      {"ACTIVATION", fmt::format("({})", actiFuncOrd)},
      {"WG_M", fmt::format("({})", m_wgTile.x)},
      {"WG_N", fmt::format("({})", m_wgTile.y)},
      {"SG_M", fmt::format("({})", m_sgTile.x)},
      {"SG_K", fmt::format("({})", m_sgTile.y)},
      {"SG_N", fmt::format("({})", m_sgTile.z)},
      {"SG_SIZE", fmt::format("({})", subgroupSize)},
      {"SG_COUNT", fmt::format("({})", subgroupCount)},
      {"USE_BIAS", op.biasType.has_value() ? "(true)" : "(false)"},
      {"ASYNC_READ", m_asyncRead ? "(true)" : "(false)"},
  };

  for (const auto &def : defines) {
    fmt::println("#define {} {}", def.name, def.value);
  }

  std::optional<WeightDescriptor::Bias> bias = std::nullopt;
  if (op.biasType.has_value()) {
    if (m_cmShape.z == 16) {
      fmt::println("BiasLayout: C16");
      bias.emplace(BiasLayout::C16, *op.biasType);
    } else if (m_cmShape.z == 8) {
      fmt::println("BiasLayout: C8");
      bias.emplace(BiasLayout::C8, *op.biasType);
    } else {
      throw std::runtime_error("Not supported");
    }
  }

  fmt::println("CM_M: {}", m_cmShape.x);
  fmt::println("CM_K: {}", m_cmShape.y);
  fmt::println("CM_N: {}", m_cmShape.z);

  fmt::println("SG_M: {}", m_sgTile.x);
  fmt::println("SG_K: {}", m_sgTile.y);
  fmt::println("SG_N: {}", m_sgTile.z);

  fmt::println("WG_M: {}", m_wgTile.x);
  fmt::println("WG_N: {}", m_wgTile.y);

  const unsigned int channelTile = m_cmShape.z * m_sgTile.z * m_wgTile.y;
  const unsigned int xtile = m_cmShape.x;
  const unsigned int ytile = m_sgTile.x * m_wgTile.x;

  std::string name(this->name());
  return ConvShaderSource(
      std::move(src), ShaderLang::GLSL, SpecializationConstants{specConstants},
      ShaderDefines{defines}, glm::uvec3(channelTile, xtile, ytile),
      op.inputLayout, op.inputType, op.outputLayout, op.outputType,
      WeightDescriptor{
          op.filterShape,
          FilterLayout::RCSKC8,
          FloatType::F16,
          bias,
      },
      op.stride, op.padding, name);
}

std::string_view ConvGEMM::name() const { return "conv"; };

} // namespace vkcnn::shaders
