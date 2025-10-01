#include "shaders/slice/MemorySliceShader.hpp"
#include "memory/dtype/dtype.hpp"

namespace denox::compiler::shaders {

MemorySliceShader::MemorySliceShader(GlslCompiler *compiler)
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
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
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

  auto dispatch = impl.registerDispatch(std::move(shader));
  dispatch.addBinding(inId);
  dispatch.addBinding(outId);
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
