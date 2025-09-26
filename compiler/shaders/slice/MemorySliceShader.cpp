#include "shaders/slice/MemorySliceShader.hpp"

namespace denox::compiler::shaders {

MemorySliceShader::MemorySliceShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  {
    Pattern p;
    auto in = p.matchNode();
    auto slice = in->matchOutgoing();
    auto out = slice->matchDst();
    slice->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Slice; });

    m_patternHandles.emplace_back(in, std::move(slice), out);
    m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                         std::move(out));
  }
}
memory::optional<unsigned int> MemorySliceShader::acceptMatch(
    [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
        &opGraph,
    unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  if (in.layout != out.layout) {
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

  auto dispatch = impl.dispatch({});
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
