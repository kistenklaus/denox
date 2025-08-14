#include "./CopyTransformShader.hpp"
#include "vkcnn/common/io/read_file.hpp"
#include "vkcnn/common/shader/CopyShaderSource.hpp"
#include "vkcnn/common/shader/ShaderDefines.hpp"
#include "vkcnn/shaders/preprocessing.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <glm/fwd.hpp>

namespace vkcnn::shaders {

static std::vector<std::byte> source() {
  std::string srcStr =
      vkcnn::readFile("./shaders/src/vkcnn/shaders/copy/copy_transform.comp");

  std::string str = vkcnn::shaders::preprocess_shader_src_pragmas(srcStr);

  std::vector<std::byte> src(str.size() * sizeof(std::string::value_type));
  std::memcpy(src.data(), str.data(), src.size());
  return src;
}

CopyTransformShader::CopyTransformShader(glm::uvec3 wTile, glm::uvec3 iTile)
    : m_tileSizes(TileSizes{wTile, iTile}), m_source(source()),
      m_name(fmt::format("copy_transform_{}x{}x{}_{}x{}x{}", wTile.x, wTile.y,
                         wTile.z, iTile.x, iTile.y, iTile.z)) {}

CopyTransformShader::CopyTransformShader()
    : m_tileSizes(std::nullopt), m_source(source()), m_name("copy_transform") {}

bool CopyTransformShader::supports(const OpCopy &op) const {
  ActivationLayout inLayout = op.inputLayout;
  if (inLayout == ActivationLayout::HWC &&
      (op.inputChannelOffset + op.channels) % 8 == 0) {
    inLayout = ActivationLayout::HWC8; // promote layout
  }
  ActivationLayout outLayout = op.outputLayout;
  if (outLayout == ActivationLayout::HWC &&
      (op.outputChannelOffset + op.channels) % 8 == 0) {
    outLayout = ActivationLayout::HWC8; // promote layout
  }

  if (inLayout != outLayout) {
    return false;
  }
  if (op.inputType != FloatType::F16) {
    return false;
  }
  if (op.outputType != FloatType::F16) {
    return false;
  }
  return true;
}

CopyShaderSource CopyTransformShader::do_specialize(const OpCopy &op) const {

  ActivationLayout inLayout = op.inputLayout;
  if (inLayout == ActivationLayout::HWC &&
      (op.inputChannelOffset + op.channels) % 8 == 0) {
    inLayout = ActivationLayout::HWC8;
  }

  std::string inLayoutMacro;
  std::string istype;
  unsigned int istype_size;
  if (inLayout == ActivationLayout::HWC) {
    inLayoutMacro = "IN_LAYOUT_HWC";
    if (op.inputType == FloatType::F16) {
      istype = "uint16_t";
      istype_size = 2;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (inLayout == ActivationLayout::HWC8) {
    inLayoutMacro = "IN_LAYOUT_HWC8";
    if (op.inputType == FloatType::F16) {
      istype = "uvec4";
      istype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (inLayout == ActivationLayout::CHW) {
    inLayoutMacro = "IN_LAYOUT_CHW";
    if (op.inputType == FloatType::F16) {
      istype = "uint16_t";
      istype_size = 2;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (inLayout == ActivationLayout::CHWC4) {
    inLayoutMacro = "IN_LAYOUT_CHWC4";
    if (op.inputType == FloatType::F16) {
      istype = "uvec2";
      istype_size = 8;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (inLayout == ActivationLayout::CHWC8) {
    inLayoutMacro = "IN_LAYOUT_CHWC8";
    if (op.inputType == FloatType::F16) {
      istype = "uvec4";
      istype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (inLayout == ActivationLayout::CHWC16) {
    inLayoutMacro = "IN_LAYOUT_CHWC16";
    if (op.inputType == FloatType::F16) {
      istype = "uvec4";
      istype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else {
    throw std::runtime_error("Unsupported layout");
  }

  ActivationLayout outLayout = op.outputLayout;
  if (outLayout == ActivationLayout::HWC &&
      (op.outputChannelOffset + op.channels) % 8 == 0) {
    outLayout = ActivationLayout::HWC8; // promote layout
  }

  std::string outLayoutMacro;
  std::string ostype;
  unsigned int ostype_size;
  if (outLayout == ActivationLayout::HWC) {
    outLayoutMacro = "OUT_LAYOUT_HWC";
    if (op.inputType == FloatType::F16) {
      ostype = "uint16_t";
      ostype_size = 2;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (outLayout == ActivationLayout::HWC8) {
    outLayoutMacro = "OUT_LAYOUT_HWC8";
    if (op.inputType == FloatType::F16) {
      ostype = "uvec4";
      ostype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (outLayout == ActivationLayout::CHW) {
    outLayoutMacro = "OUT_LAYOUT_CHW";
    if (op.inputType == FloatType::F16) {
      ostype = "uint16_t";
      ostype_size = 2;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (outLayout == ActivationLayout::CHWC4) {
    outLayoutMacro = "OUT_LAYOUT_CHWC4";
    if (op.inputType == FloatType::F16) {
      ostype = "uvec2";
      ostype_size = 8;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (outLayout == ActivationLayout::CHWC8) {
    outLayoutMacro = "OUT_LAYOUT_CHWC8";
    if (op.inputType == FloatType::F16) {
      ostype = "uvec4";
      ostype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else if (outLayout == ActivationLayout::CHWC16) {
    outLayoutMacro = "OUT_LAYOUT_CHWC16";
    if (op.inputType == FloatType::F16) {
      ostype = "uvec4";
      ostype_size = 16;
    } else {
      throw std::runtime_error("Not supported");
    }
  } else {
    throw std::runtime_error("Unsupported layout");
  }

  std::string in_atype;
  unsigned int in_atype_size;
  if (op.inputType == FloatType::F16) {
    in_atype = "float16_t";
    in_atype_size = 2;
  } else {
    throw std::runtime_error("Not supported");
  }

  std::string out_atype;
  unsigned int out_atype_size;
  if (op.outputType == FloatType::F16) {
    out_atype = "float16_t";
    out_atype_size = 2;
  } else {
    throw std::runtime_error("Not supported");
  }

  const unsigned int subgroupSize = 32;

  glm::uvec3 wTile;
  glm::uvec3 iTile;
  if (m_tileSizes.has_value()) {
    wTile = m_tileSizes->wTile;
    iTile = m_tileSizes->iTile;
  } else {
    // NOTE: autotune based on layouts and channel count. (currently just a big
    // switch statement)
    //
    // TODO: A heuristic here would be nice, but not 100% sure what
    // hardware stat is should actually be dependent on.
    if (inLayout == ActivationLayout::HWC) {
      // NOTE: With HWC (i.e. 2 byte accesses) we try to spread all channel
      // loads across invocations.
      if (op.channels >= 16) {
        iTile = glm::uvec3(2, 2, 1);
        wTile = glm::uvec3(8, 32, 1);
      } else {
        iTile = glm::uvec3(1, 4, 1);
        wTile = glm::uvec3(op.channels, 32, 1);
      }
    } else if (inLayout == ActivationLayout::HWC8) {
      if (op.channels >= 32) {
        iTile = glm::uvec3(8, 1, 1);
        wTile = glm::uvec3(4, 64, 1);
      } else if (op.channels >= 16) {
        iTile = glm::uvec3(8, 1, 1);
        wTile = glm::uvec3(2, 128, 1);
      } else {
        iTile = glm::uvec3(8, 1, 1);
        wTile = glm::uvec3(1, 256, 1);
      }
    } else if (inLayout == ActivationLayout::CHWC8) {
      iTile = glm::uvec3(8, 1, 1);
      wTile = glm::uvec3(1, 256, 1);
    } else {
      throw std::runtime_error("Autotuning is not implemented");
    }
  }

  glm::uvec3 tileSize = wTile * iTile;

  ShaderDefine defines[22] = {
      {"SG_SIZE", fmt::format("({})", subgroupSize)}, //
      {"WG_C", fmt::format("({})", wTile.x)},         //
      {"WG_W", fmt::format("({})", wTile.y)},         //
      {"WG_H", fmt::format("({})", wTile.z)},         //
      {"INVOC_C", fmt::format("({})", iTile.x)},      //
      {"INVOC_W", fmt::format("({})", iTile.y)},      //
      {"INVOC_H", fmt::format("({})", iTile.z)},      //
                                                      //
      {inLayoutMacro, "(1)"},
      {"istype", istype},                                //
      {"ISTYPE_SIZE", fmt::format("({})", istype_size)}, //
                                                         //
      {outLayoutMacro, "(1)"},
      {"ostype", ostype},                                             //
      {"OSTYPE_SIZE", fmt::format("({})", ostype_size)},              //
                                                                      //
      {"CH", fmt::format("({})", op.channels)},                       //
      {"IN_CH_OFFSET", fmt::format("({})", op.inputChannelOffset)},   //
      {"OUT_CH_OFFSET", fmt::format("({})", op.outputChannelOffset)}, //
      {"IN_CH", fmt::format("({})", op.inputChannels)},               //
      {"OUT_CH", fmt::format("({})", op.outputChannels)},             //
                                                                      //
      {"in_atype", in_atype},                                         //
      {"IN_ATYPE_SIZE", fmt::format("({})", in_atype_size)},          //
                                                                      //
      {"out_atype", out_atype},                                       //
      {"OUT_ATYPE_SIZE", fmt::format("({})", out_atype_size)},        //
  };

  for (const ShaderDefine &def : defines) {
    fmt::println("#define {} {}", def.name, def.value);
  }

  return CopyShaderSource(
      m_source, ShaderLang::GLSL, SpecializationConstants{},
      ShaderDefines{defines}, tileSize,
      CopyShaderSource::DebugInfo(op.channels, op.inputLayout, op.inputType,
                                  op.outputLayout, op.outputType,
                                  op.inputChannelOffset, op.outputChannelOffset,
                                  op.inputChannels, op.outputChannels, m_name));
}

std::string_view CopyTransformShader::name() const { return m_name; }

} // namespace vkcnn::shaders
