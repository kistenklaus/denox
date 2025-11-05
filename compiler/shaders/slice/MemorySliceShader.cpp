#include "shaders/slice/MemorySliceShader.hpp"
#include "memory/dtype/dtype.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

MemorySliceShader::MemorySliceShader(GlslCompiler *compiler)
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
    auto slice = in->matchOutgoing();
    auto out = slice->matchDst();
    slice->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Slice; });

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    m_patternHandles.emplace_back(in, std::move(slice), out);
    m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                         std::move(out));
  }
}
memory::optional<unsigned int> MemorySliceShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  if (memory::ActivationLayout::demote(in.layout, in.channels) !=
      memory::ActivationLayout::demote(out.layout, out.channels)) {
    return memory::nullopt;
  }
  return pattern;
}
void MemorySliceShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match, SymGraph& symGraph) const {
  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  memory::EdgeId opId = match[patternHandle.slice];

  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto& slice = opGraph.get(opId).slice();

  auto shader = m_compiler->read(m_srcPath);
  assert(in.channels == out.channels);
  shader.define("CH", in.channels);

  std::uint32_t invocC;
  std::uint32_t invocW;
  std::uint32_t invocH;
  std::uint32_t wgC;
  std::uint32_t wgW;
  std::uint32_t wgH;
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
  } else {
    throw std::runtime_error("not supported");
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

  auto dispatch = impl.registerDispatch(std::move(shader),
      workgroupCountX, workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);

  dispatch.addPushConstant(PushConstant::Dynamic(slice->top));
  dispatch.addPushConstant(PushConstant::Dynamic(slice->left));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.addPushConstant(PushConstant::Dynamic(out.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(out.extent.y));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}

memory::string
MemorySliceShader::name([[maybe_unused]] unsigned int pattern) const {
  return "memory-slice";
}
} // namespace denox::compiler::shaders
