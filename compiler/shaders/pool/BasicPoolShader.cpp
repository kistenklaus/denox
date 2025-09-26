#include "shaders/pool/BasicPoolShader.hpp"
#include "model/PoolFunction.hpp"

namespace denox::compiler::shaders {

BasicPoolShader::BasicPoolShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    return tensor.layout == memory::ActivationLayout::HWC ||
           tensor.layout == memory::ActivationLayout::HWC8 ||
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
memory::optional<unsigned int> BasicPoolShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (in.layout != out.layout) {
    return memory::nullopt;
  }
  return pattern;
}
void BasicPoolShader::implement(
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
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}
memory::string BasicPoolShader::name(unsigned int) const {
  return "basic-pool";
}
} // namespace denox::compiler::shaders
