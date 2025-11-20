#include "shaders/upsample/BasicUpsampleShader.hpp"
#include "model/FilterMode.hpp"
#include <cassert>

namespace denox::compiler::shaders {

BasicUpsampleShader::BasicUpsampleShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    return tensor.layout == memory::ActivationLayout::HWC ||
           tensor.layout == memory::ActivationLayout::CHWC8;
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
      if (op.tag() != ComputeOpTag::Upsample) {
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
memory::optional<unsigned int> BasicUpsampleShader::acceptMatch(
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
    return memory::nullopt;
  }

  if (in.type != memory::Dtype::F16) {
    return memory::nullopt;
  }
  if (out.type != memory::Dtype::F16) {
    return memory::nullopt;
  }

  return pattern;
}
void BasicUpsampleShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId upsampleId = match[patternHandles.upsample];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &upsample = opGraph.get(upsampleId);

  memory::ActivationLayout inLayout = in.layout;
  memory::ActivationLayout outLayout = out.layout;

  auto shader = m_compiler->read(m_srcPath);
  shader.enableDenoxPreprocessor();

  std::uint32_t invocC;
  std::uint32_t invocW;
  std::uint32_t invocH;
  std::uint32_t wgC;
  std::uint32_t wgW;
  std::uint32_t wgH;

  assert(in.channels == out.channels);

  if (inLayout == memory::ActivationLayout::HWC &&
      outLayout == memory::ActivationLayout::HWC && (in.channels % 8 != 0)) {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
    if (in.channels >= 16) {
      invocC = 2;
      invocW = 2;
      invocH = 1;
      wgC = 8;
      wgW = 32;
      wgH = 1;
    } else {
      invocC = 1;
      invocW = 4;
      invocH = 1;
      wgC = in.channels;
      wgW = 32;
      wgH = 1;
    }
  } else if (inLayout == memory::ActivationLayout::HWC &&
             outLayout == memory::ActivationLayout::HWC &&
             (in.channels % 8 == 0 && out.channels % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");

    if (in.channels >= 32) {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 4;
      wgW = 64;
      wgH = 1;
    } else if (in.channels >= 16) {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 2;
      wgW = 128;
      wgH = 1;
    } else {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 1;
      wgW = 256;
      wgH = 1;
    }
  } else if (inLayout == memory::ActivationLayout::CHWC8 &&
             outLayout == memory::ActivationLayout::CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
    {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 1;
      wgW = 256;
      wgH = 1;
    }
  } else {
    throw std::logic_error("Invalid state");
  }
  shader.define("INVOC_C", invocC);
  shader.define("INVOC_W", invocW);
  shader.define("INVOC_H", invocH);
  shader.define("WG_C", wgC);
  shader.define("WG_W", wgW);
  shader.define("WG_H", wgH);

  assert(in.channels == out.channels);
  shader.define("CH", in.channels);
  shader.define("SCALING_FACTOR", upsample.upsample().scalingFactor);

  std::uint32_t tileX = invocC * wgC;
  std::uint32_t tileY = invocW * wgW;
  std::uint32_t tileZ = invocH * wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.extent.x.asSym(), tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.extent.y.asSym(), tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.extent.x.asSym(), in.extent.y.asSym(),
                           in.channels * in.type.size());
  Sym writes = symGraph.mul(out.extent.x.asSym(), out.extent.y.asSym(),
                            out.channels * out.type.size());
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setDebugInfo(fmt::format("BasicUpsampleShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.layout.to_string(),
                                    out.layout.to_string()));

  dispatch.setInputDesc(fmt::format("{}[{}]", in.layout.to_string(), in.channels));
  dispatch.setOutputDesc(fmt::format("{}[{}]", out.layout.to_string(), out.channels));
}
memory::string
BasicUpsampleShader::name([[maybe_unused]] unsigned int pattern) const {
  return "basic-upsample";
}
} // namespace denox::compiler::shaders
