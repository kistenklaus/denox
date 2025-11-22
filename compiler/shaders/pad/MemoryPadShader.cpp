#include "shaders/pad/MemoryPadShader.hpp"
#include "memory/dtype/dtype.hpp"

namespace denox::compiler::shaders {

struct Config {
  unsigned int invocC;
  unsigned int invocW;
  unsigned int invocH;
  memory::optional<unsigned int> wgC;
  unsigned int wgW;
  unsigned int wgH;
};

static constexpr std::array<Config, 5> CONFIGS{
    Config{
        .invocC = 2,
        .invocW = 2,
        .invocH = 2,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    Config{
        .invocC = 1,
        .invocW = 4,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },
};

MemoryPadShader::MemoryPadShader(GlslCompiler *compiler)
    : m_compiler(compiler) {

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::HWC) {
      return false;
    }
    return true;
  };
  {
    Pattern p;
    auto in = p.matchNode();
    auto pad = in->matchOutgoing();
    auto out = pad->matchDst();

    pad->matchRank(1);
    pad->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Pad; });

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    m_patternHandles.emplace_back(in, std::move(pad), out);
    m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                         std::move(out));
  }
}
memory::vector<unsigned int> MemoryPadShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];

  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  if (memory::ActivationLayout::demote(in.layout, in.channels) !=
      memory::ActivationLayout::demote(out.layout, out.channels)) {
    return {};
  }
  return {0, 1, 2, 3, 4};
}

static GlslCompilerInstance compile(GlslCompiler *compiler,
                                    const io::Path &srcPath,
                                    memory::ActivationLayout inputLayout,
                                    memory::ActivationLayout outputLayout,
                                    unsigned int channels,
                                    const Config &config) {
  auto shader = compiler->read(srcPath);
  shader.define("CH", channels);

  if (inputLayout == memory::ActivationLayout::HWC &&
      outputLayout == memory::ActivationLayout::HWC && (channels % 8 != 0)) {
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
    compiler::diag::unreachable();
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC.value_or(channels));
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);
  return shader;
}

void MemoryPadShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const Config &config = CONFIGS[configKey];
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &padId = match[patternHandles.pad];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const ComputeOpPad &pad = opGraph.get(padId).pad();

  assert(in.channels == out.channels);
  auto shader = compile(m_compiler, m_srcPath, in.layout, out.layout,
                        in.channels, config);

  std::uint32_t tileX = config.invocC * config.wgC.value_or(in.channels);
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.extent.x.asSym(), tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.extent.y.asSym(), tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);

  dispatch.addPushConstant( //
      PushConstant::Dynamic(out.extent.x, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(out.extent.y, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->left, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->right, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->top, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->bottom, memory::Dtype::U32));

  dispatch.setName(name(pattern, 0));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.extent.x.asSym(), in.extent.y.asSym(),
                           in.channels * in.type.size());
  Sym writes = symGraph.mul(out.extent.x.asSym(), out.extent.y.asSym(),
                            out.channels * out.type.size());
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setDebugInfo(fmt::format("MemoryPadShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.layout.to_string(),
                                    out.layout.to_string()));

  dispatch.setInputDesc(
      fmt::format("{}[{}]", in.layout.to_string(), in.channels));
  dispatch.setOutputDesc(
      fmt::format("{}[{}]", out.layout.to_string(), out.channels));
}
memory::string MemoryPadShader::name(unsigned int, unsigned int) const {
  return "memory-pad";
}
} // namespace denox::compiler::shaders
