#include "shaders/pad/MemoryPadShader.hpp"

namespace denox::compiler::shaders {

MemoryPadShader::MemoryPadShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  {
    Pattern p;
    auto in = p.matchNode();
    auto pad = in->matchOutgoing();
    auto out = pad->matchDst();

    pad->matchRank(1);
    pad->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Pad; });

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
  if (in.layout != out.layout) {
    return memory::nullopt;
  }
  return pattern;
}
void MemoryPadShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];

  auto dispatch = impl.dispatch({});
  dispatch.addBinding(inId);
  dispatch.addBinding(outId);

  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
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
