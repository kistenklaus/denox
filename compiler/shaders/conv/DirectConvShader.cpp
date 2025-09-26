#include "shaders/conv/DirectConvShader.hpp"
#include "diag/unreachable.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"

namespace denox::compiler::shaders {

DirectConvShader::DirectConvShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto tensorSupported = [](const TensorInstance &tensor) {
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
    Pattern conv_pattern;
    auto in = conv_pattern.matchNode();
    auto conv = in->matchOutgoing();
    conv->matchRank(1);
    conv->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Conv; });
    auto out = conv->matchDst();

    in->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_patternHandles.emplace_back(in, std::move(conv), memory::nullopt, out);
    m_capabilities.patterns.emplace_back(std::move(conv_pattern), std::move(in),
                                         std::move(out));
  }
  { // possibly more patterns.
    Pattern conv_relu_pattern;
    auto in = conv_relu_pattern.matchNode();
    auto conv = in->matchOutgoing();
    auto inter = conv->matchDst();
    auto relu = inter->matchOutgoing();
    auto out = relu->matchDst();

    conv->matchRank(1);
    conv->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Conv; });
    relu->matchRank(1);
    relu->matchValue([](const ComputeOp &op) {
      return op.tag() == ComputeOpTag::Activation;
    });

    in->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_patternHandles.emplace_back(in, std::move(conv), std::move(relu), out);
    m_capabilities.patterns.emplace_back(std::move(conv_relu_pattern),
                                         std::move(in), std::move(out));
  }
}
std::size_t DirectConvShader::parameterMemorySize(
    const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &convPattern = m_patternHandles[pattern].conv;
  memory::EdgeId convId = match[convPattern];
  const ComputeOp &op = graph.get(convId);
  assert(op.tag() == ComputeOpTag::Conv);
  const auto &conv = op.conv();
  std::size_t elemCount = conv->W->shape().elemCount();
  if (conv->B != nullptr) {
    elemCount += conv->B->shape();
  }
  return elemCount * memory::Dtype::F16.size();
}
void DirectConvShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::EdgeId convId = match[patternHandles.conv];
  const ComputeOp &op = opGraph.get(convId);
  assert(op.tag() == ComputeOpTag::Conv);
  const ComputeOpConv &conv = op.conv();
  TensorId weightTensorId = impl.createParameter(*conv->W);
  memory::optional<TensorId> biasTensorId = memory::nullopt;
  if (conv->B != nullptr) {
    biasTensorId = impl.createParameter(*conv->B);
  }

  auto dispatch = impl.dispatch({});
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  dispatch.addBinding(inId);
  dispatch.addBinding(outId);
  dispatch.addBinding(weightTensorId);
  if (biasTensorId) {
    dispatch.addBinding(*biasTensorId);
  }
  auto in = opGraph.get(inId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}
memory::string DirectConvShader::name(unsigned int pattern) const {
  switch (pattern) {
  case CONV_PATTERN:
    return "direct-conv";
  case CONV_RELU_PATTERN:
    return "direct-conv+relu";
  default:
    compiler::diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
