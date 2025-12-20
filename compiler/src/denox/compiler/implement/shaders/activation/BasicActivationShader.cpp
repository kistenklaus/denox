#include "shaders/activation/BasicActivationShader.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "model/ActivationFunction.hpp"

namespace denox::compiler::shaders {

struct BasicActivationConfig {
  std::uint32_t invocC;
  std::uint32_t invocW;
  std::uint32_t invocH;
  memory::optional<std::uint32_t> wgC;
  std::uint32_t wgW;
  std::uint32_t wgH;
};

static constexpr std::array<BasicActivationConfig, 5> CONFIGS = {
    BasicActivationConfig{
        .invocC = 2,
        .invocW = 2,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    BasicActivationConfig{
        .invocC = 1,
        .invocW = 4,
        .invocH = 1,
        .wgC = memory::nullopt, // <- insert channel count.
        .wgW = 32,
        .wgH = 1,
    },
    BasicActivationConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    BasicActivationConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    BasicActivationConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    }};

BasicActivationShader::BasicActivationShader(GlslCompiler *compiler,
                                             const Options &options)
    : m_compiler(compiler),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize)

{
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::HWC &&
        tensor.layout != memory::ActivationLayout::CHWC8) {
      return false;
    }
    return true;
  };
  {
    Pattern basicPattern;
    auto in = basicPattern.matchNode();
    auto acti = in->matchOutgoing();
    auto out = acti->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    acti->matchRank(1);
    acti->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpTag::Activation) {
        return false;
      }
      if (op.activation().func != ActivationFunction::ReLU) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(acti), out);
    m_capabilities.patterns.emplace_back(std::move(basicPattern), std::move(in),
                                         std::move(out));
  }
}

memory::vector<unsigned int> BasicActivationShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (memory::ActivationLayout::demote(in.layout, in.channels) !=
      memory::ActivationLayout::demote(out.layout, out.channels)) {
    return {};
  }
  auto layout = memory::ActivationLayout::promote(in.layout, in.channels);
  std::vector<unsigned int> configs;
  configs.reserve(CONFIGS.size());
  for (unsigned int c = 0; c < CONFIGS.size(); ++c) {
    const auto& config = CONFIGS[c];
    uint32_t wgC = CONFIGS[c].wgC.value_or(in.channels);
    uint32_t workGroupInvocations = wgC * config.wgH * config.wgW;
    if (workGroupInvocations >= m_maxComputeWorkGroupInvocations) {
      continue;
    }
    if (wgC >= m_maxComputeWorkGroupSize[0]) {
      continue;
    }
    if (config.wgW >= m_maxComputeWorkGroupSize[1]) {
      continue;
    }
    if (config.wgH >= m_maxComputeWorkGroupSize[2]) {
      continue;
    }
    if (config.invocC % layout.vectorBlockSize() != 0) {
      continue;
    }
    configs.push_back(c);
  }
  return configs;
}

static GlslCompilerInstance
compile(GlslCompiler *compiler, const io::Path &srcPath,
        unsigned int subgroupSize, memory::ActivationLayout inputLayout,
        memory::ActivationLayout outputLayout, unsigned int channels,
        memory::Dtype atype, ActivationFunction activationFunction,
        const BasicActivationConfig &config) {
  if (atype != memory::Dtype::F16) {
    diag::invalid_state();
  }
  auto shader = compiler->read(srcPath);
  shader.define("SG_SIZE", subgroupSize);
  shader.define("CH", channels);
  shader.define("in_atype", "float16_t");
  shader.define("IN_ATYPE_SIZE", 2);
  shader.define("out_atype", "float16_t");
  shader.define("OUT_ATYPE_SIZE", 2);

  switch (activationFunction) {
  case ActivationFunction::ReLU:
    shader.define("ACTIVATION_ReLU");
    break;
  case ActivationFunction::LeakyReLU:
  case ActivationFunction::SiLU:
    diag::invalid_state();
  }
  if (inputLayout == memory::ActivationLayout::HWC &&
      outputLayout == memory::ActivationLayout::HWC && (channels % 8 != 0)) {
    // HWC layout (slow path)
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);

    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
  } else if (inputLayout == memory::ActivationLayout::HWC &&
             outputLayout == memory::ActivationLayout::HWC &&
             (channels % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");
  } else if (inputLayout == memory::ActivationLayout::CHWC8 &&
             outputLayout == memory::ActivationLayout::CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC.value_or(channels));
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  return shader;
}

void BasicActivationShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const BasicActivationConfig &config = CONFIGS[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &acti = opGraph.get(match[patternHandles.acti]).activation();

  assert(in.channels == out.channels);
  auto shader =
      compile(m_compiler, m_srcPath, m_subgroupSize, in.layout, out.layout,
              in.channels, memory::Dtype::F16, acti.func, config);

  std::uint32_t tileC = config.invocC * config.wgC.value_or(in.channels);
  std::uint32_t tileW = config.invocW * config.wgW;
  std::uint32_t tileH = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(in.channels, tileC);
  Sym workgroupCountY = symGraph.cdiv(in.extent.x.asSym(), tileW);
  Sym workgroupCountZ = symGraph.cdiv(in.extent.y.asSym(), tileH);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setSourcePath(m_srcPath);

  Sym reads =
      symGraph.mul(symGraph.mul(in.extent.x.asSym(), in.extent.y.asSym()),
                   in.channels * in.type.size());
  dispatch.setMemoryReads(reads);
  Sym writes =
      symGraph.mul(symGraph.mul(out.extent.x.asSym(), out.extent.y.asSym()),
                   out.channels * out.type.size());
  dispatch.setMemoryWrites(writes);

  switch (acti.func) {
  case ActivationFunction::ReLU:
    dispatch.setName("relu");
    dispatch.setDebugInfo(fmt::format("{}-relu-{}", in.layout.to_string(),
                                      out.layout.to_string()));
    break;
  case ActivationFunction::LeakyReLU:
    dispatch.setName("leaky-relu");
    dispatch.setDebugInfo(fmt::format("{}-leaky-relu-{}", in.layout.to_string(),
                                      out.layout.to_string()));
    break;
  case ActivationFunction::SiLU:
    dispatch.setName("silu");
    dispatch.setDebugInfo(fmt::format("{}-silu-{}", in.layout.to_string(),
                                      out.layout.to_string()));
    break;
    break;
  }
  dispatch.setInputDesc(
      fmt::format("{}[{}]", in.layout.to_string(), in.channels));
  dispatch.setOutputDesc(
      fmt::format("{}[{}]", out.layout.to_string(), out.channels));
}

memory::string BasicActivationShader::name(unsigned int, unsigned int) const {
  return "basic-activation";
}
} // namespace denox::compiler::shaders
