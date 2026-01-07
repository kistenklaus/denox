#include "denox/compiler/implement/shaders/pool/BasicPoolShader.hpp"
#include "denox/common/PoolFunction.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

struct BasicPoolConfig {
  const unsigned int cdiv;
  const memory::optional<unsigned int> invocC;
  const unsigned int invocW;
  const unsigned int invocH;
  const unsigned int wgC;
  const unsigned int wgW;
  const unsigned int wgH;
};

static std::array<BasicPoolConfig, 5> CONFIGS = {
    BasicPoolConfig{
        .cdiv = 4,
        .invocC = memory::nullopt,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    BasicPoolConfig{
        .cdiv = 8,
        .invocC = memory::nullopt,
        .invocW = 1,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    BasicPoolConfig{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    BasicPoolConfig{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    BasicPoolConfig{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },
};

BasicPoolShader::BasicPoolShader(spirv::GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != TensorDataType::Float16) {
      return false;
    }
    if (tensor.storage != TensorStorage::StorageBuffer) {
      return false;
    }
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    return tensor.format == TensorFormat::SSBO_HWC ||
           tensor.format == TensorFormat::SSBO_CHWC8;
  };

  {

    Pattern poolPattern;
    auto in = poolPattern.matchNode();
    auto pool = in->matchOutgoing();
    auto out = pool->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    pool->matchRank(1);

    pool->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Pool) {
        return false;
      }
      const auto &pool = op.pool();
      if (pool->func != PoolFunction::Max) {
        return false;
      }
      if (pool->stride != pool->kernelSize) {
        return false;
      }
      if (pool->padding != memory::uvec2(0, 0)) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(pool), out);
    m_capabilities.patterns.emplace_back(std::move(poolPattern), std::move(in),
                                         std::move(out));
  }
}
memory::vector<unsigned int> BasicPoolShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);

  if (in.format != out.format) {
    return {};
  }
  if (in.type != TensorDataType::Float16) {
    return {};
  }
  if (out.type != TensorDataType::Float16) {
    return {};
  }

  uint32_t cblocksize;
  switch (in.format) {
  case TensorFormat::SSBO_HWC:
    if (in.channels.constant() % 8 == 0) {
      cblocksize = 8;
    } else {
      cblocksize = 1;
    }
    break;
  case TensorFormat::SSBO_CHW:
    cblocksize = 1;
    break;
  case TensorFormat::SSBO_CHWC8:
    cblocksize = 8;
    break;
  case TensorFormat::Optimal:
  case TensorFormat::TEX_RGBA:
  case TensorFormat::TEX_RGB:
  case TensorFormat::TEX_RG:
  case TensorFormat::TEX_R:
    diag::invalid_state();
    break;
  }

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  memory::vector<unsigned int> configs;
  configs.reserve(CONFIGS.size());
  for (unsigned int c = 0; c < CONFIGS.size(); ++c) {
    unsigned int invocC;
    if (CONFIGS[c].invocC.has_value()) {
      invocC = CONFIGS[c].invocC.value();
    } else {
      assert(CONFIGS[c].cdiv != 0);
      invocC = (C + CONFIGS[c].cdiv - 1) / CONFIGS[c].cdiv;
    }
    if (invocC % cblocksize == 0) {
      configs.push_back(c);
    }
  }
  return configs;
}

static spirv::GlslCompilerInstance
compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
        TensorFormat inputFormat, TensorFormat outputFormat,
        unsigned int channels, memory::uvec2 kernelSize, memory::uvec2 stride,
        memory::uvec2 padding, const BasicPoolConfig &config) {
  auto shader = compiler->read(srcPath);

  uint32_t invocC;
  if (config.invocC) {
    invocC = *config.invocC;
  } else {
    const unsigned int ix = (channels + config.cdiv - 1) / config.cdiv;
    invocC = ix;
  }

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC && (channels % 8 == 0) &&
      (invocC % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");
  } else if (inputFormat == TensorFormat::SSBO_HWC &&
             outputFormat == TensorFormat::SSBO_HWC) {
    if (channels % 8 == 0) {
      DENOX_WARN(
          "BasicPoolShader implements non vectorized layouts for format, "
          "which may be vectorized, this works, but is suboptimal!");
    }
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
  } else if (inputFormat == TensorFormat::SSBO_CHWC8 &&
             outputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    throw std::logic_error("Invalid state");
  }

  shader.define("INVOC_C", invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  shader.define("CH", channels);

  shader.define("KERNEL_X", kernelSize.x);
  shader.define("KERNEL_Y", kernelSize.y);
  shader.define("STRIDE_X", stride.x);
  shader.define("STRIDE_Y", stride.y);
  shader.define("PADDING_X", padding.x);
  shader.define("PADDING_Y", padding.y);
  return shader;
}

void BasicPoolShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const BasicPoolConfig &config = CONFIGS[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId poolId = match[patternHandles.pool];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &pool = opGraph.get(poolId).pool();

  assert(in.channels == out.channels);

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  auto shader = compile(m_compiler, m_srcPath, in.format, out.format, C,
                        pool->kernelSize, pool->stride, pool->padding, config);

  unsigned int invocC;
  if (config.invocC) {
    invocC = *config.invocC;
  } else {
    invocC = (C + config.cdiv - 1) / config.cdiv;
  }

  std::uint32_t tileX = invocC * config.wgC;
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, Access::ReadOnly, inId);
  dispatch.addBinding(0, 1, Access::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.width));
  dispatch.addPushConstant(PushConstant::Dynamic(in.height));
  dispatch.setName(name(pattern, 0));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setDebugInfo(fmt::format("BasicPoolShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.format, out.format));
}
memory::string BasicPoolShader::name(unsigned int, unsigned int) const {
  return "basic-pool";
}
} // namespace denox::compiler::shaders
