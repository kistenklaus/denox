#include "shaders/pad/MemoryPadShader.hpp"
#include "memory/dtype/dtype.hpp"

namespace denox::compiler::shaders {

MemoryPadShader::MemoryPadShader(GlslCompiler *compiler)
    : m_compiler(compiler) {

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
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);

  auto shader = m_compiler->read(m_srcPath);

  if (in.layout == memory::ActivationLayout::HWC &&
      out.layout == memory::ActivationLayout::HWC &&
      (in.channels % 8 != 0 || out.channels % 8 != 0)) {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);

    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");

    {
      shader.define("WG_C", 32);
      shader.define("WG_W", 8);
      shader.define("WG_H", 1);
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

    {
      shader.define("WG_C", 32);
      shader.define("WG_W", 8);
      shader.define("WG_H", 1);
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
      shader.define("WG_C", 32);
      shader.define("WG_W", 8);
      shader.define("WG_H", 1);
    }
  }

  Sym workgroupCountX = Sym::Const(1);
  Sym workgroupCountY = Sym::Const(1);
  Sym workgroupCountZ = Sym::Const(1);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);

  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.addPushConstant(PushConstant::Dynamic(out.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(out.extent.y));

  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}
memory::string MemoryPadShader::name(unsigned int) const {
  return "memory-pad";
}
} // namespace denox::compiler::shaders
