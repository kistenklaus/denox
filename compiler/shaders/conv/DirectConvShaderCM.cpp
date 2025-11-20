#include "shaders/conv/DirectConvShaderCM.hpp"
#include "Options.hpp"
#include "diag/invalid_state.hpp"
#include "diag/unreachable.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "memory/tensor/BiasDescriptor.hpp"
#include "memory/tensor/FilterTensor.hpp"
#include "memory/tensor/FitlerDescriptor.hpp"
#include "model/ActivationFunction.hpp"

namespace denox::compiler::shaders {

DirectConvShaderCM::DirectConvShaderCM(GlslCompiler *compiler,
                                       const Options &options)
    : m_compiler(compiler),
      m_enableConvReluFusion(options.fusionRules.enableConvReluFusion),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize) {

  if (m_subgroupSize == 0) {
    return;
  }
  if (options.features.coopmat == FeatureState::Disable) {
    return;
  }
  if (options.deviceInfo.coopmat.supported == false) {
    return;
  }

  const auto tensorSupported = [](const TensorInstance &tensor) {
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
  if (m_enableConvReluFusion) { // possibly more patterns.
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
      if (op.tag() != ComputeOpTag::Activation) {
        return false;
      }
      if (op.activation().func != ActivationFunction::ReLU) {
        return false;
      }
      return true;
    });

    in->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_patternHandles.emplace_back(in, std::move(conv), std::move(relu), out);
    m_capabilities.patterns.emplace_back(std::move(conv_relu_pattern),
                                         std::move(in), std::move(out));
  }
}
std::size_t DirectConvShaderCM::parameterMemorySize(
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
void DirectConvShaderCM::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern,
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
  assert(op.tag() == ComputeOpTag::Conv);
  const ComputeOpConv &conv = op.conv();
  auto shader = m_compiler->read(m_srcPath);

  if (in.channels % 8 == 0) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
  } else {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
  }

  if (out.channels % 8 == 0) {
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
  } else {
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
  }

  if (in.layout == memory::ActivationLayout::HWC && in.channels % 8 == 0) {
    shader.define("IN_LAYOUT_HWC8");
  } else if (in.layout == memory::ActivationLayout::HWC &&
             in.channels % 8 != 0) {
    shader.define("IN_LAYOUT_HWC");
  } else if (in.layout == memory::ActivationLayout::CHWC8) {
    shader.define("IN_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  if (out.layout == memory::ActivationLayout::HWC && out.channels % 8 == 0) {
    shader.define("OUT_LAYOUT_HWC8");
  } else if (out.layout == memory::ActivationLayout::HWC &&
             out.channels % 8 != 0) {
    shader.define("OUT_LAYOUT_HWC");
  } else if (out.layout == memory::ActivationLayout::CHWC8) {
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  if (pattern == CONV_ACTIVATION_PATTERN) {
    const auto &activation =
        opGraph.get(match[*m_patternHandles[pattern].relu]).activation();
    switch (activation.func) {
    case ActivationFunction::ReLU:
      shader.define("ACTIVATION_ReLU");
      break;
    case ActivationFunction::LeakyReLU:
    case ActivationFunction::SiLU:
      diag::invalid_state();
      break;
    }
  } else {
    shader.define("ACTIVATION_NONE");
  }
  // TODO Properly select coopmat shape.
  unsigned int cm_m = 16;
  unsigned int cm_k = 8;
  unsigned int cm_n = 8;
  shader.define("CM_M", cm_m);
  shader.define("CM_K", cm_k);
  shader.define("CM_N", cm_n);

  unsigned int wg_m = 8;
  unsigned int wg_n = 1;
  shader.define("WG_M", wg_m);
  shader.define("WG_N", wg_n);
  unsigned int subgroupCount = wg_n * wg_m;
  shader.define("SG_COUNT", subgroupCount);

  unsigned int sg_m = 2;
  unsigned int sg_k = 2;
  unsigned int sg_n = 2;
  shader.define("SG_M", sg_m);
  shader.define("SG_K", sg_k);
  shader.define("SG_N", sg_n);

  memory::FilterLayout filterLayout = memory::FilterLayout::RSCK;
  if (in.channels % cm_k == 0 && (cm_k == 8 || cm_k == 16)) {
    if (cm_k == 8) {
      filterLayout = memory::FilterLayout::RSCKC8;
      shader.define("FILTER_LAYOUT_RSCKC8");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else if (cm_k == 16) {
      filterLayout = memory::FilterLayout::RSCKC16;
      shader.define("FILTER_LAYOUT_RSCKC16");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else if (out.channels % cm_n == 0 && (cm_n == 8 || cm_n == 16)) {
    if (cm_n == 8) {
      filterLayout = memory::FilterLayout::KRSCK8;
      shader.define("FILTER_LAYOUT_KRSCK8");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else if (cm_n == 16) {
      filterLayout = memory::FilterLayout::KRSCK16;
      shader.define("FILTER_LAYOUT_KRSCK16");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else {
    filterLayout = memory::FilterLayout::RSCK;
    shader.define("FILTER_LAYOUT_RSCK");
    shader.define("fstype", "uint16_t");
    shader.define("FSTYPE_SIZE", 2);
  }

  shader.define("ASYNC_READ");
  // if (in.channels >= 128 || out.channels >= 128) {
  // } else {
  //   shader.define("NASYNC_READ");
  // }

  shader.define("atype", "float16_t");
  shader.define("ATYPE_SIZE", 2);
  shader.define("IN_CH", in.channels);
  shader.define("OUT_CH", out.channels);

  shader.define("SG_SIZE", m_subgroupSize);
  shader.define("KERNEL_X", conv->W->shape().s);
  shader.define("KERNEL_Y", conv->W->shape().r);
  shader.define("STRIDE_X", conv->stride.x);
  shader.define("STRIDE_Y", conv->stride.y);
  shader.define("PADDING_X", conv->padding.x);
  shader.define("PADDING_Y", conv->padding.y);

  if (conv->B != nullptr) {
    shader.define("USE_BIAS");
  } else {
    shader.define("NUSE_BIAS");
  }

  std::uint32_t tileX = cm_n * sg_n * wg_n;
  std::uint32_t tileY = cm_m;
  std::uint32_t tileZ = sg_m * wg_m;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(in.extent.x.asSym(), tileY);
  Sym workgroupCountZ = symGraph.cdiv(in.extent.y.asSym(), tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  // Convert to expected layout!
  memory::FilterTensor filterWeights{
      memory::FilterDescriptor{
          .shape = conv->W->shape(),
          .layout = filterLayout,
          .type = memory::Dtype::F16,
      },
      memory::FilterTensorConstView(conv->W.get())};

  TensorId weightTensorId = impl.createParameter(filterWeights);
  memory::optional<TensorId> biasTensorId = memory::nullopt;

  if (conv->B != nullptr) {
    memory::BiasLayout biasLayout = memory::BiasLayout::C16;
    if (cm_n == 8) {
      biasLayout = memory::BiasLayout::C8;
    } else if (cm_n == 16) {
      biasLayout = memory::BiasLayout::C16;
    }
    memory::BiasTensor biasWeights{memory::BiasDescriptor{
                                       .shape = conv->B->shape(),
                                       .layout = biasLayout,
                                       .type = memory::Dtype::F16,
                                   },
                                   memory::BiasTensorConstView(conv->B.get())};
    biasTensorId = impl.createParameter(biasWeights);
  }

  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addBinding(0, 2, AccessFlag::ReadOnly, weightTensorId);
  if (biasTensorId) {
    dispatch.addBinding(0, 3, AccessFlag::ReadOnly, *biasTensorId);
  }
  dispatch.addPushConstant(
      PushConstant::Dynamic(in.extent.x, memory::Dtype::U32));
  dispatch.addPushConstant(
      PushConstant::Dynamic(in.extent.y, memory::Dtype::U32));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);

  Sym inreads =
      symGraph.mul(symGraph.mul(in.extent.x.asSym(), in.extent.y.asSym()),
                   in.channels * in.type.size());
  size_t wreads = conv->W->byteSize() + (conv->B ? conv->B->byteSize() : 0ull);
  Sym reads = symGraph.add(wreads, inreads);
  Sym writes =
      symGraph.mul(symGraph.mul(out.extent.x.asSym(), out.extent.y.asSym()),
                   in.channels * in.type.size());
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);

  dispatch.setDebugInfo(fmt::format("{}-direct-conv-{}", in.layout.to_string(),
                                    out.layout.to_string()));

  dispatch.setInputDesc(fmt::format("{}[{}]", in.layout.to_string(), in.channels));
  dispatch.setOutputDesc(fmt::format("{}[{}]", out.layout.to_string(), out.channels));
}
memory::string DirectConvShaderCM::name(unsigned int pattern) const {
  switch (pattern) {
  case CONV_PATTERN:
    return "direct-conv";
  case CONV_ACTIVATION_PATTERN:
    return "direct-conv+activation";
  default:
    compiler::diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
