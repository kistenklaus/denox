#include "denox/compiler/implement/shaders/conv/DirectConvShaderCM.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include <fmt/format.h>

namespace denox::compiler::shaders {

struct DirectConvConfigCM {
  unsigned int cm_m;
  unsigned int cm_k;
  unsigned int cm_n;
  unsigned int wg_m;
  unsigned int wg_n;
  unsigned int sg_m;
  unsigned int sg_k;
  unsigned int sg_n;
  bool async;
};

static std::vector<DirectConvConfigCM> DIRECT_CONV_CM_CONFIGS = {
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 2,
        .sg_k = 2,
        .sg_n = 2,
        .async = true,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 2,
        .sg_k = 2,
        .sg_n = 2,
        .async = false,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 4,
        .wg_n = 2,

        .sg_m = 2,
        .sg_k = 2,
        .sg_n = 2,
        .async = true,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 4,
        .wg_n = 2,

        .sg_m = 2,
        .sg_k = 2,
        .sg_n = 2,
        .async = false,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 1,
        .sg_n = 6,
        .async = true,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 1,
        .sg_n = 6,
        .async = false,
    },

    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 1,
        .sg_n = 7,
        .async = true,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 1,
        .sg_n = 7,
        .async = false,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 3,
        .sg_n = 6,
        .async = true,
    },
    DirectConvConfigCM{
        .cm_m = 16,
        .cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 1,
        .sg_k = 3,
        .sg_n = 7,
        .async = false,
    },

    // // 16x8x8 coop shape.
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 4,
    //     .wg_n = 2,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 4,
    //     .wg_n = 2,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 6,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 6,
    //     .async = false,
    // },
    //
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 7,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 7,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 6,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 8,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 7,
    //     .async = false,
    // },

    // // 16x8x16 coopmat shape
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 4,
    //     .wg_n = 2,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 4,
    //     .wg_n = 2,
    //
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 6,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 6,
    //     .async = false,
    // },
    //
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 7,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 7,
    //     .async = false,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 6,
    //     .async = true,
    // },
    // DirectConvConfigCM{
    //     .cm_m = 16,
    //     .cm_k = 8,
    //     .cm_n = 16,
    //
    //     .wg_m = 8,
    //     .wg_n = 1,
    //
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 7,
    //     .async = false,
    // },
};

DirectConvShaderCM::DirectConvShaderCM(spirv::GlslCompiler *compiler,
                                       const CompileOptions &options)
    : m_compiler(compiler),
      m_enableConvReluFusion(options.features.enableConvReluFusion),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize),
      m_supportedCoopmatShapes(options.deviceInfo.coopmat.shapes) {

  if (m_subgroupSize == 0) {
    return;
  }
  if (!options.features.coopmat) {
    return;
  }
  if (options.deviceInfo.coopmat.supported == false) {
    return;
  }

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
std::size_t DirectConvShaderCM::parameterMemorySize(
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

memory::vector<unsigned int> DirectConvShaderCM::acceptMatch(
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

  std::vector<unsigned int> configs;
  configs.reserve(DIRECT_CONV_CM_CONFIGS.size());
  for (unsigned int c = 0; c < DIRECT_CONV_CM_CONFIGS.size(); ++c) {
    const auto &config = DIRECT_CONV_CM_CONFIGS[c];
    uint32_t sgCount = config.wg_n * config.wg_m;
    if (sgCount > m_maxComputeWorkGroupSize[1]) {
      continue;
    }
    uint32_t workgroupInvocations = sgCount * m_subgroupSize;
    if (workgroupInvocations > m_maxComputeWorkGroupInvocations) {
      continue;
    }
    bool coopmatShapeSupported = false;
    for (const auto &shape : m_supportedCoopmatShapes) {
      if (shape.subgroupScope && shape.atype == memory::Dtype::F16 &&
          shape.btype == memory::Dtype::F16 &&
          shape.ctype == memory::Dtype::F16 &&
          shape.acctype == memory::Dtype::F16 && shape.M == config.cm_m &&
          shape.K == config.cm_k && shape.N == config.cm_n) {
        coopmatShapeSupported = true;
        break;
      }
    }
    if (!coopmatShapeSupported) {
      continue;
    }

    configs.push_back(c);
  }
  return configs;
}

static spirv::GlslCompilerInstance direct_conv_cm_compile(
    spirv::GlslCompiler *compiler, const io::Path &srcPath,
    unsigned int subgroupSize, unsigned int C, unsigned int K,
    TensorFormat inputFormat, TensorFormat outputFormat,
    memory::optional<ActivationFunction> activationFunction,
    memory::uvec2 kernelSize, memory::uvec2 padding, memory::uvec2 stride,
    bool bias, const DirectConvConfigCM &config,
    //
    memory::FilterLayout *out_filterLayout,
    memory::BiasLayout *out_biasLayout) {
  auto shader = compiler->read(srcPath);
  if (C % 8 == 0) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
  } else {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
  }

  if (K % 8 == 0) {
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
  } else {
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
  }

  if (inputFormat == TensorFormat::SSBO_HWC && C % 8 == 0) {
    shader.define("IN_LAYOUT_HWC8");
  } else if (inputFormat == TensorFormat::SSBO_HWC && C % 8 != 0) {
    shader.define("IN_LAYOUT_HWC");
  } else if (inputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("IN_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  if (outputFormat == TensorFormat::SSBO_HWC && K % 8 == 0) {
    shader.define("OUT_LAYOUT_HWC8");
  } else if (outputFormat == TensorFormat::SSBO_HWC && K % 8 != 0) {
    shader.define("OUT_LAYOUT_HWC");
  } else if (outputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }
  if (activationFunction) {
    switch (*activationFunction) {
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

  memory::FilterLayout filterLayout = memory::FilterLayout::RSCK;
  if ((C % config.cm_k == 0) && ((config.cm_k == 8) || (config.cm_k == 16))) {
    if (config.cm_k == 8) {
      filterLayout = memory::FilterLayout::RSCKC8;
      shader.define("FILTER_LAYOUT_RSCKC8");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else if (config.cm_k == 16) {
      filterLayout = memory::FilterLayout::RSCKC16;
      shader.define("FILTER_LAYOUT_RSCKC16");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else if (K % config.cm_n == 0 && (config.cm_n == 8 || config.cm_n == 16)) {
    if (config.cm_n == 8) {
      filterLayout = memory::FilterLayout::KRSCK8;
      shader.define("FILTER_LAYOUT_KRSCK8");
      shader.define("fstype", "uvec4");
      shader.define("FSTYPE_SIZE", 16);
    } else if (config.cm_n == 16) {
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
  if (config.async) {
    shader.define("ASYNC_READ");
  } else {
    shader.define("NASYNC_READ");
  }
  *out_filterLayout = filterLayout;
  // if (in.channels >= 128 || out.channels >= 128) {
  // } else {
  //   shader.define("NASYNC_READ");
  // }

  shader.define("atype", "float16_t");
  shader.define("ATYPE_SIZE", 2);
  shader.define("IN_CH", C);
  shader.define("OUT_CH", K);

  shader.define("SG_SIZE", subgroupSize);
  unsigned int subgroupCount = config.wg_n * config.wg_m;
  shader.define("SG_COUNT", subgroupCount);

  shader.define("KERNEL_X", kernelSize.x);
  shader.define("KERNEL_Y", kernelSize.y);
  shader.define("STRIDE_X", stride.x);
  shader.define("STRIDE_Y", stride.y);
  shader.define("PADDING_X", padding.x);
  shader.define("PADDING_Y", padding.y);

  shader.define("SG_M", config.sg_m);
  shader.define("SG_K", config.sg_k);
  shader.define("SG_N", config.sg_n);

  shader.define("WG_M", config.wg_m);
  shader.define("WG_N", config.wg_n);

  shader.define("CM_M", config.cm_m);
  shader.define("CM_K", config.cm_k);
  shader.define("CM_N", config.cm_n);

  if (bias) {
    shader.define("USE_BIAS");
    if (config.cm_n == 8) {
      *out_biasLayout = memory::BiasLayout::C8;
    } else if (config.cm_n == 16) {
      *out_biasLayout = memory::BiasLayout::C16;
    } else {
      *out_biasLayout = memory::BiasLayout::C;
    }

  } else {
    shader.define("NUSE_BIAS");
  }
  return shader;
}

void DirectConvShaderCM::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {
  const DirectConvConfigCM &config = DIRECT_CONV_CM_CONFIGS[configKey];

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

  memory::FilterLayout filterLayout = memory::FilterLayout::KCRS;
  memory::BiasLayout biasLayout = memory::BiasLayout::C;
  auto shader = direct_conv_cm_compile(
      m_compiler, m_srcPath, m_subgroupSize, C, K, in.format, out.format,
      activationFunction, memory::uvec2(conv->W->shape().r, conv->W->shape().s),
      conv->padding, conv->stride, conv->B != nullptr, config, //
      &filterLayout, &biasLayout);

  std::uint32_t tileX = config.cm_n * config.sg_n * config.wg_n;
  std::uint32_t tileY = config.cm_m;
  std::uint32_t tileZ = config.sg_m * config.wg_m;

  // tileX = 16 * 2 * 1 = 32
  // tileY = 16
  // tileZ = 2 * 8 = 16

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX); // 8 / 16 = 1
  Sym workgroupCountY = symGraph.cdiv(in.width, tileY);     // 1920 / 16 = 120
  Sym workgroupCountZ = symGraph.cdiv(in.height, tileZ);    // 1080 / 16 = 68

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);

  TensorId weightTensorId = impl.createParameter(
      filterLayout.size(conv->W->shape()) * memory::Dtype::F16.size(),
      TensorDataType::Float16, TensorStorage::StorageBuffer,
      TensorFormat::Optimal,
      [W = conv->W, filterLayout]() -> std::vector<std::byte> {
        memory::FilterTensor filter{
            {W->shape(), filterLayout, memory::Dtype::F16}, W->const_view()};
        std::vector<std::byte> raw(filter.span().begin(), filter.span().end());
        return raw;
      });

  memory::optional<TensorId> biasTensorId = memory::nullopt;

  if (conv->B != nullptr) {
    biasTensorId = impl.createParameter(
        biasLayout.size(conv->B->shape()) * memory::Dtype::F16.size(),
        TensorDataType::Float16, TensorStorage::StorageBuffer,
        TensorFormat::Optimal,
        [B = conv->B, biasLayout]() -> std::vector<std::byte> {
          memory::BiasTensor bias{{B->shape(), biasLayout, memory::Dtype::F16},
                                  B->const_view()};
          std::vector<std::byte> raw(bias.span().begin(), bias.span().end());
          return raw;
        });
  }

  dispatch.addBinding("INPUT_SET", "INPUT_BINDING", Access::ReadOnly, inId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addBinding("FILTER_SET", "FILTER_BINDING", Access::ReadOnly,
                      weightTensorId);
  if (biasTensorId) {
    dispatch.addBinding("BIAS_SET", "BIAS_BINDING", Access::ReadOnly,
                        *biasTensorId);
  }

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
memory::string DirectConvShaderCM::name(unsigned int pattern,
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
