#include "denox/compiler/implement/shaders/upsample/BasicUpsampleShader.hpp"
#include "denox/common/FilterMode.hpp"
#include "denox/diag/invalid_state.hpp"
#include <cassert>

namespace denox::compiler::shaders {

struct BasicUpsampleConfig {
  unsigned int invocC;
  unsigned int invocW;
  unsigned int invocH;
  memory::optional<unsigned int> wgC;
  unsigned int wgW;
  unsigned int wgH;
};

static std::array<BasicUpsampleConfig, 5> CONFIGS{
    BasicUpsampleConfig{
        // not supported
        .invocC = 2,
        .invocW = 2,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    BasicUpsampleConfig{
        // not supported
        .invocC = 1,
        .invocW = 4,
        .invocH = 1,
        .wgC = memory::nullopt,
        .wgW = 32,
        .wgH = 1,
    },
    BasicUpsampleConfig{
        // 0.383ms
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    BasicUpsampleConfig{
        // 0.319ms
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    BasicUpsampleConfig{
        // 0.318ms
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },

};

BasicUpsampleShader::BasicUpsampleShader(spirv::GlslCompiler *compiler,
                                         const CompileOptions &options)
    : m_compiler(compiler),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != TensorDataType::Float16) {
      return false;
    }
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    if (tensor.storage != TensorStorage::StorageBuffer) {
      return false;
    }

    return tensor.format == TensorFormat::SSBO_HWC ||
           tensor.format == TensorFormat::SSBO_CHWC8;
  };
  {
    Pattern upsamplePattern;
    auto in = upsamplePattern.matchNode();
    auto upsample = in->matchOutgoing();
    auto out = upsample->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      const auto &upsample = op.upsample();
      if (upsample.mode != FilterMode::Nearest) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(upsample), out);
    m_capabilities.patterns.emplace_back(std::move(upsamplePattern),
                                         std::move(in), std::move(out));
  }
}
memory::vector<unsigned int> BasicUpsampleShader::acceptMatch(
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
    diag::invalid_state();
  }
  if (out.type != TensorDataType::Float16) {
    diag::invalid_state();
  }

  if (in.channels != out.channels) {
    diag::invalid_state();
  }
  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  uint32_t cblocksize;
  switch (in.format) {
  case TensorFormat::SSBO_HWC:
    if (in.channels.constant() % 8 == 0) {
      cblocksize = 8;
    } else {
      cblocksize = 1;
    }
    break;
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
  }

  memory::vector<unsigned int> configs;
  configs.reserve(CONFIGS.size());
  for (unsigned int c = 0; c < CONFIGS.size(); ++c) {
    const auto &config = CONFIGS[c];
    uint32_t wgC = config.wgC.value_or(C);
    uint32_t workgroupInvocations = wgC * config.wgW * config.wgH;
    if (workgroupInvocations > m_maxComputeWorkGroupInvocations) {
      continue;
    }
    if (wgC > m_maxComputeWorkGroupSize[0]) {
      continue;
    }
    if (config.wgW > m_maxComputeWorkGroupSize[1]) {
      continue;
    }
    if (config.wgH > m_maxComputeWorkGroupSize[2]) {
      continue;
    }
    if (config.invocC % cblocksize != 0) {
      continue;
    }
    configs.push_back(c);
  }
  return configs;
}

static spirv::GlslCompilerInstance
compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
        TensorFormat inputFormat, TensorFormat outputFormat,
        unsigned int channels, unsigned int scalingFactor,
        const BasicUpsampleConfig &config) {
  auto shader = compiler->read(srcPath);
  shader.enableDenoxPreprocessor();

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC &&
      (channels % 8 == 0 && config.invocC % 8 == 0)) {
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
          "BasicUpsampleShader implements non vectorized layouts for format, "
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

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC.value_or(channels));
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  shader.define("CH", channels);
  shader.define("SCALING_FACTOR", scalingFactor);
  return shader;
}

void BasicUpsampleShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {
  const BasicUpsampleConfig config = CONFIGS[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId upsampleId = match[patternHandles.upsample];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &upsample = opGraph.get(upsampleId);

  uint32_t C = static_cast<uint32_t>(in.channels.constant());
  assert(C == out.channels.constant());

  auto shader = compile(m_compiler, m_srcPath, in.format, out.format, C,
                        upsample.upsample().scalingFactor, config);

  std::uint32_t tileX = config.invocC * config.wgC.value_or(C);
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
  dispatch.setDebugInfo(fmt::format("BasicUpsampleShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.format, out.format));
}
memory::string BasicUpsampleShader::name([[maybe_unused]] unsigned int pattern,
                                         unsigned int) const {
  return "basic-upsample";
}
} // namespace denox::compiler::shaders
