#include "./DirectPoolShader.hpp"
#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/io/read_file.hpp"
#include "vkcnn/common/ops/OpPool.hpp"
#include "vkcnn/common/shader/PoolShaderSource.hpp"
#include "vkcnn/shaders/preprocessing.hpp"
#include <cstring>
#include <fmt/format.h>

namespace vkcnn::shaders {

static std::vector<std::byte> source() {
  std::string srcStr =
      vkcnn::readFile("./shaders/src/vkcnn/shaders/pool/direct_pool.comp");

  std::string str = vkcnn::shaders::preprocess_shader_src_pragmas(srcStr);

  std::vector<std::byte> src(str.size() * sizeof(std::string::value_type));
  std::memcpy(src.data(), str.data(), src.size());
  return src;
}

DirectPoolShader::DirectPoolShader(glm::uvec3 iTile, glm::uvec3 wTile)
    : m_tileSizes(TileSizes(wTile, iTile)), m_source(source()),
      m_name(fmt::format("direct_pool_W{}x{}x{}_I{}x{}x{}", wTile.x, wTile.y,
                         wTile.z, iTile.x, iTile.y, iTile.z)) {}

DirectPoolShader::DirectPoolShader()
    : m_tileSizes(std::nullopt), m_source(source()), m_name("direct_pool") {}

bool DirectPoolShader::supports(const OpPool &op) const {
  if (op.inputLayout != op.outputLayout) {
    return false;
  }
  if (op.inputType != op.outputType) {
    return false;
  }
  if (op.poolingFunc != PoolFunction::Max) {
    return false;
  }
  return true;
}

PoolShaderSource DirectPoolShader::do_specialize(const OpPool &op) const {
  ActivationLayout inLayout = op.inputLayout;
  if (inLayout == ActivationLayout::HWC && op.channels % 8 == 0) {
    // inLayout = ActivationLayout::HWC8;
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
  if (outLayout == ActivationLayout::HWC && op.channels % 8 == 0) {
    // outLayout = ActivationLayout::HWC8; // promote layout
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
      if (op.channels <= 24) {
        unsigned int ix = (op.channels + 4 - 1) / 4;
        iTile = glm::uvec3(ix, 1, 1);
        wTile = glm::uvec3(4, 64, 1);
      } else if (op.channels <= 48) {
        unsigned int ix = (op.channels + 8 - 1) / 8;
        iTile = glm::uvec3(ix, 1, 1);
        wTile = glm::uvec3(8, 32, 1);
      } else {
        wTile = glm::uvec3(16, 16, 1);
        iTile = glm::uvec3(2, 2, 1);
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

  std::string poolFuncMacro;
  if (op.poolingFunc == PoolFunction::Max) {
    poolFuncMacro = "POOL_FUNC_MAX";
  } else {
    throw std::runtime_error("Not supported");
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

  ShaderDefine defines[25] = {
      {"SG_SIZE", fmt::format("({})", subgroupSize)}, //

      {"WG_C", fmt::format("({})", wTile.x)}, //
      {"WG_W", fmt::format("({})", wTile.y)}, //
      {"WG_H", fmt::format("({})", wTile.z)}, //

      {"INVOC_C", fmt::format("({})", iTile.x)}, //
      {"INVOC_W", fmt::format("({})", iTile.y)}, //
      {"INVOC_H", fmt::format("({})", iTile.z)}, //

      {inLayoutMacro, "(1)"},
      {"istype", istype},                                //
      {"ISTYPE_SIZE", fmt::format("({})", istype_size)}, //

      {outLayoutMacro, "(1)"},
      {"ostype", ostype},                                //
      {"OSTYPE_SIZE", fmt::format("({})", ostype_size)}, //

      {"in_atype", in_atype},                                //
      {"IN_ATYPE_SIZE", fmt::format("({})", in_atype_size)}, //

      {"out_atype", out_atype},                                //
      {"OUT_ATYPE_SIZE", fmt::format("({})", out_atype_size)}, //

      {"CH", fmt::format("({})", op.channels)}, //

      {poolFuncMacro, "(1)"}, //

      {"KERNEL_X", fmt::format("({})", op.kernelSize.x)}, //
      {"KERNEL_Y", fmt::format("({})", op.kernelSize.y)}, //
      {"STRIDE_X", fmt::format("({})", op.stride.x)},     //
      {"STRIDE_Y", fmt::format("({})", op.stride.y)},     //
      {"PADDING_X", fmt::format("({})", op.padding.x)},   //
      {"PADDING_Y", fmt::format("({})", op.padding.y)},   //

  };
  for (const ShaderDefine &def : defines) {
    fmt::println("#define {} {}", def.name, def.value);
  }

  glm::uvec3 tileSize = wTile * iTile;

  return PoolShaderSource(m_source, ShaderLang::GLSL, SpecializationConstants{},
                          ShaderDefines{defines}, tileSize,
                          PoolShaderSource::DebugInfo(
                              op.inputLayout, op.inputType, op.outputLayout,
                              op.outputType, op.channels, op.kernelSize,
                              op.stride, op.padding, op.poolingFunc, m_name));
}

std::string_view DirectPoolShader::name() const { return m_name; }

} // namespace vkcnn::shaders
