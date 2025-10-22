#include "shaders/pad/MemoryPadShader.hpp"
#include "memory/dtype/dtype.hpp"
#include <fmt/base.h>

namespace denox::compiler::shaders {

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
memory::optional<unsigned int> MemoryPadShader::acceptMatch(
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
    return memory::nullopt;
  }
  return pattern;
}
void MemoryPadShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &padId = match[patternHandles.pad];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const ComputeOpPad &pad = opGraph.get(padId).pad();

  auto shader = m_compiler->read(m_srcPath);

  std::uint32_t invocC;
  std::uint32_t invocW;
  std::uint32_t invocH;
  std::uint32_t wgC;
  std::uint32_t wgW;
  std::uint32_t wgH;

  shader.define("CH", in.channels);
  assert(out.channels == in.channels);
  if (in.layout == memory::ActivationLayout::HWC &&
      out.layout == memory::ActivationLayout::HWC &&
      (in.channels % 8 != 0 || out.channels % 8 != 0)) {
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
  } else if (in.layout == memory::ActivationLayout::HWC &&
             out.layout == memory::ActivationLayout::HWC &&
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
    } else if (out.channels >= 16) {
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
  } else if (in.layout == memory::ActivationLayout::CHWC8 &&
             out.layout == memory::ActivationLayout::CHWC8) {
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
    compiler::diag::unreachable();
  }

  shader.define("INVOC_C", invocC);
  shader.define("INVOC_W", invocW);
  shader.define("INVOC_H", invocH);
  shader.define("WG_C", wgC);
  shader.define("WG_W", wgW);
  shader.define("WG_H", wgH);

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

  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}
memory::string MemoryPadShader::name(unsigned int) const {
  return "memory-pad";
}
} // namespace denox::compiler::shaders
