#include "denox/common/ActivationFunction.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/conv/DirectConvShader.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include <fmt/format.h>

namespace denox::compiler::shaders {

DirectConvShader::DirectConvShader(spirv::GlslCompiler *compiler,
                                       const CompileOptions &options)
    : m_compiler(compiler),
      m_enableConvReluFusion(options.features.enableConvReluFusion),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize) {

  const auto tensorSupported = [](const TensorInstance &tensor) {
    if (tensor.type != TensorDataType::Float16) {
      return false;
    }
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    if (tensor.storage != TensorStorage::StorageBuffer) {
      return false;
    }
    if (tensor.format != TensorFormat::SSBO_HWC &&
        tensor.format != TensorFormat::SSBO_CHWC8) {
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
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::Conv; });
    auto out = conv->matchDst();

    in->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_patternHandles.emplace_back(in, std::move(conv), memory::nullopt, out);
    m_capabilities.patterns.emplace_back(std::move(conv_pattern), std::move(in),
                                         std::move(out));
  }
  if (m_enableConvReluFusion) { // possibly more patterns.
    Pattern conv_relu_pattern;
    auto in = conv_relu_pattern.matchNode();
    auto conv = in->matchOutgoing();
    auto inter = conv->matchDst();
    auto relu = inter->matchOutgoing();
    auto out = relu->matchDst();

    conv->matchRank(1);
    conv->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::Conv; });
    relu->matchRank(1);
    relu->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      if (op.activation().func != ActivationFunction::ReLU) {
        return false;
      }
      return true;
    });

    in->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_patternHandles.emplace_back(in, conv, relu, out);
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
  assert(op.tag() == ComputeOpKind::Conv);
  const auto &conv = op.conv();
  std::size_t elemCount = conv->W->shape().elemCount();
  if (conv->B != nullptr) {
    elemCount += conv->B->shape();
  }
  return elemCount * memory::Dtype::F16.size();
}

memory::vector<unsigned int> DirectConvShader::acceptMatch(
    [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
        &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  // const auto &patternHandles = m_patternHandles[pattern];
  // const auto &in = opGraph.get(match[patternHandles.in]);
  // const auto &out = opGraph.get(match[patternHandles.out]);

  if (m_subgroupSize > m_maxComputeWorkGroupSize[0]) {
    return {};
  }
  return {0};
  ;
}

void DirectConvShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {

  const auto &patternHandles = m_patternHandles[pattern];
  memory::EdgeId convId = match[patternHandles.conv];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const ComputeOp &op = opGraph.get(convId);
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  assert(op.tag() == ComputeOpKind::Conv);
  assert(in.channels.isConstant());
  assert(out.channels.isConstant());
  const ComputeOpConv &conv = op.conv();

  memory::optional<ActivationFunction> activationFunction;

  if (pattern == CONV_ACTIVATION_PATTERN) {
    activationFunction =
        opGraph.get(match[*m_patternHandles[pattern].relu]).activation().func;
  }

  uint32_t C = static_cast<uint32_t>(in.channels.constant());
  uint32_t K = static_cast<uint32_t>(out.channels.constant());

  Sym workgroupCountX = Sym::Const(1);
  Sym workgroupCountY = Sym::Const(1);
  Sym workgroupCountZ = Sym::Const(1);

  auto shader = m_compiler->read(m_srcPath);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  // Convert to expected layout!
  // memory::FilterTensor filterWeights{
  //     memory::FilterDescriptor{
  //         .shape = conv->W->shape(),
  //         .layout = filterLayout,
  //         .type = memory::Dtype::F16,
  //     },
  //     memory::FilterTensorConstView(conv->W.get())};

  dispatch.addBinding(0, 0, Access::ReadOnly, inId);
  dispatch.addBinding(0, 1, Access::WriteOnly, outId);

  dispatch.addPushConstant(PushConstant::Dynamic(in.width, memory::Dtype::U32));
  dispatch.addPushConstant(
      PushConstant::Dynamic(in.height, memory::Dtype::U32));
  dispatch.setName(name(pattern, configKey));
  dispatch.setSourcePath(m_srcPath);

  Sym inreads =
      symGraph.mul(symGraph.mul(in.width, in.height), C * size_of(in.type));
  size_t wreads = conv->W->byteSize() + (conv->B ? conv->B->byteSize() : 0ull);
  Sym reads = symGraph.add(wreads, inreads);
  Sym writes =
      symGraph.mul(symGraph.mul(out.width, out.height), K * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);

  dispatch.setDebugInfo(
      fmt::format("{}-direct-conv-{}", in.format, out.format));
}
memory::string DirectConvShader::name(unsigned int pattern,
                                        unsigned int) const {
  switch (pattern) {
  case CONV_PATTERN:
    return "direct-conv-cm";
  case CONV_ACTIVATION_PATTERN:
    return "direct-conv-cm+activation";
  default:
    diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
