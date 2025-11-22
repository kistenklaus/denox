#include "shaders/pool/BasicPoolShader.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/PoolFunction.hpp"
#include "shaders/compiler/GlslCompilerInstance.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

struct Config {
  unsigned int cdiv;
  memory::optional<unsigned int> invocC;
  unsigned int invocW;
  unsigned int invocH;
  unsigned int wgC;
  unsigned int wgW;
  unsigned int wgH;
};

static constexpr std::array<Config, 5> CONFIGS{
    Config{
        .cdiv = 4,
        .invocC = memory::nullopt,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    Config{
        .cdiv = 8,
        .invocC = memory::nullopt,
        .invocW = 1,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    Config{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    Config{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    Config{
        .cdiv = 0,
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },
};

BasicPoolShader::BasicPoolShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    return tensor.layout == memory::ActivationLayout::HWC ||
           tensor.layout == memory::ActivationLayout::CHWC8;
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
      if (op.tag() != ComputeOpTag::Pool) {
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

  memory::ActivationLayout inLayout = in.layout;
  memory::ActivationLayout outLayout = out.layout;

  if (memory::ActivationLayout::demote(outLayout, out.channels) !=
      memory::ActivationLayout::demote(inLayout, in.channels)) {
    return {};
  }
  if (in.type != memory::Dtype::F16) {
    return {};
  }
  if (out.type != memory::Dtype::F16) {
    return {};
  }
  return {0, 1, 2, 3, 4};
}

static GlslCompilerInstance
compile(GlslCompiler *compiler, const io::Path &srcPath,
        memory::ActivationLayout inputLayout,
        memory::ActivationLayout outputLayout, unsigned int channels,
        memory::uvec2 kernelSize, memory::uvec2 stride, memory::uvec2 padding,
        const Config &config) {
  auto shader = compiler->read(srcPath);

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
    throw std::logic_error("Invalid state");
  }

  if (config.invocC) {
    shader.define("INVOC_C", *config.invocC);
  } else {
    assert(config.cdiv != 0);
    unsigned int ix = (channels + config.cdiv - 1) / config.cdiv;
    shader.define("INVOC_C", ix);
  }
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
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const Config &config = CONFIGS[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId poolId = match[patternHandles.pool];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &pool = opGraph.get(poolId).pool();

  assert(in.channels == out.channels);
  auto shader =
      compile(m_compiler, m_srcPath, in.layout, out.layout, in.channels,
              pool->kernelSize, pool->stride, pool->padding, config);

  unsigned int invocC;
  if (config.invocC) {
    invocC = *config.invocC;
  } else {
    invocC = (in.channels + config.cdiv - 1) / config.cdiv;
  }

  std::uint32_t tileX = invocC * config.wgC;
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.extent.x.asSym(), tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.extent.y.asSym(), tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(pattern, 0));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.extent.x.asSym(), in.extent.y.asSym(),
                           in.channels * in.type.size());
  Sym writes = symGraph.mul(out.extent.x.asSym(), out.extent.y.asSym(),
                            out.channels * out.type.size());
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setDebugInfo(fmt::format("BasicPoolShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.layout.to_string(),
                                    out.layout.to_string()));

  dispatch.setInputDesc(
      fmt::format("{}[{}]", in.layout.to_string(), in.channels));
  dispatch.setOutputDesc(
      fmt::format("{}[{}]", out.layout.to_string(), out.channels));
}
memory::string BasicPoolShader::name(unsigned int, unsigned int) const {
  return "basic-pool";
}
} // namespace denox::compiler::shaders
