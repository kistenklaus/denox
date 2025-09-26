#include "shaders/activation/BasicActivationShader.hpp"
#include "diag/unreachable.hpp"
#include "memory/dtype/dtype.hpp"
#include "model/ActivationFunction.hpp"

namespace denox::compiler::shaders {

BasicActivationShader::BasicActivationShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::HWC &&
        tensor.layout != memory::ActivationLayout::HWC8 &&
        tensor.layout != memory::ActivationLayout::CHWC8) {
      return false;
    }
    return true;
  };
  {
    Pattern basicPattern;
    auto in = basicPattern.matchNode();
    auto acti = in->matchOutgoing();
    auto out = acti->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    acti->matchRank(1);
    acti->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpTag::Activation) {
        return false;
      }
      if (op.activation().func != ActivationFunction::ReLU) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(acti), out);
    m_capabilities.patterns.emplace_back(std::move(basicPattern), std::move(in),
                                         std::move(out));
  }
}
memory::optional<unsigned int> BasicActivationShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (in.layout != out.layout) {
    return memory::nullopt;
  }

  return pattern | ACTI_FUNC_TYPE_ReLU;
}
void BasicActivationShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int patternEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  unsigned int pattern = patternEnc & ~ACTI_FUNC_TYPE_MASK;
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);

  auto dispatch = impl.dispatch({});
  dispatch.addBinding(inId);
  dispatch.addBinding(outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(patternEnc));
  dispatch.setSourcePath(m_srcPath);
}
memory::string BasicActivationShader::name(unsigned int pattern) const {
  switch (pattern & ACTI_FUNC_TYPE_MASK) {
  case ACTI_FUNC_TYPE_ReLU:
    return "relu";
  default:
    compiler::diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
