#include "denox/compiler/implement/shaders/conv/ConcatConvCMShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasDescriptor.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FitlerDescriptor.hpp"
#include <fmt/format.h>

namespace denox::compiler::shaders {

struct ConcatConvConfig {
  unsigned int cm_m;
  unsigned int a_cm_k;
  unsigned int b_cm_k;
  unsigned int cm_n;
  unsigned int wg_m;
  unsigned int wg_n;
  unsigned int sg_m;
  unsigned int a_sg_k;
  unsigned int b_sg_k;
  unsigned int sg_n;
  bool a_async;
  bool b_async;
};

static std::vector<ConcatConvConfig> CONCAT_CONV_CM_CONFIGS = {
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,

        .wg_m = 8,
        .wg_n = 1,

        .sg_m = 2,
        .a_sg_k = 2,
        .b_sg_k = 2,
        .sg_n = 2,

        .a_async = true,
        .b_async = true,
    },
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 8,
        .wg_n = 1,
    
        .sg_m = 2,
        .a_sg_k = 2,
        .b_sg_k = 2,
        .sg_n = 2,
    
        .a_async = false,
        .b_async = false,
    },
    
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 4,
        .wg_n = 2,
    
        .sg_m = 2,
        .a_sg_k = 2,
        .b_sg_k = 2,
        .sg_n = 2,
    
        .a_async = true,
        .b_async = true,
    },
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 4,
        .wg_n = 2,
    
        .sg_m = 2,
        .a_sg_k = 2,
        .b_sg_k = 2,
        .sg_n = 2,
    
        .a_async = false,
        .b_async = false,
    },
    
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 8,
        .wg_n = 1,
    
        .sg_m = 1,
        .a_sg_k = 1,
        .b_sg_k = 1,
        .sg_n = 6,
    
        .a_async = true,
        .b_async = true,
    },
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 8,
        .wg_n = 1,
    
        .sg_m = 1,
        .a_sg_k = 1,
        .b_sg_k = 1,
        .sg_n = 6,
    
        .a_async = false,
        .b_async = false,
    },
    
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 8,
        .wg_n = 1,
    
        .sg_m = 1,
        .a_sg_k = 3,
        .b_sg_k = 3,
        .sg_n = 7,
    
        .a_async = true,
        .b_async = true,
    },
    ConcatConvConfig{
        .cm_m = 16,
        .a_cm_k = 16,
        .b_cm_k = 16,
        .cm_n = 16,
    
        .wg_m = 8,
        .wg_n = 1,
    
        .sg_m = 1,
        .a_sg_k = 3,
        .b_sg_k = 3,
        .sg_n = 7,
    
        .a_async = false,
        .b_async = false,
    },
};

ConcatConvCMShader::ConcatConvCMShader(spirv::GlslCompiler *compiler,
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
    Pattern concat_conv_pattern;
    auto concat = concat_conv_pattern.matchEdge();
    auto a = concat->matchSrc(0);
    auto b = concat->matchSrc(1);
    auto x = concat->matchDst();
    auto conv = x->matchOutgoing();
    auto out = conv->matchDst();

    concat->matchRank(2);
    concat->matchValue([](const ComputeOp &op) -> bool {
      return op.tag() == ComputeOpKind::Concat;
    });
    conv->matchValue([](const ComputeOp &op) -> bool {
      return op.tag() == ComputeOpKind::Conv;
    });
    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);
    m_patternHandles.emplace_back(a, b, concat, conv, memory::nullopt, out);
    m_capabilities.patterns.emplace_back(std::move(concat_conv_pattern),
                                         std::move(a), std::move(b),
                                         std::move(out));
  }
  if (options.features.enableConvReluFusion) {
    Pattern concat_conv_pattern;
    auto concat = concat_conv_pattern.matchEdge();
    auto a = concat->matchSrc(0);
    auto b = concat->matchSrc(1);
    auto x = concat->matchDst();
    auto conv = x->matchOutgoing();
    auto y = conv->matchDst();
    auto relu = y->matchOutgoing();
    auto out = relu->matchDst();

    concat->matchRank(2);
    concat->matchValue([](const ComputeOp &op) -> bool {
      return op.tag() == ComputeOpKind::Concat;
    });
    conv->matchValue([](const ComputeOp &op) -> bool {
      return op.tag() == ComputeOpKind::Conv;
    });
    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);
    m_patternHandles.emplace_back(a, b, std::move(concat), std::move(conv),
                                  std::move(relu), out);
    m_capabilities.patterns.emplace_back(std::move(concat_conv_pattern),
                                         std::move(a), std::move(b),
                                         std::move(out));
  }
}
std::size_t ConcatConvCMShader::parameterMemorySize(
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

memory::vector<unsigned int> ConcatConvCMShader::acceptMatch(
    [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
        &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  [[maybe_unused]] const auto &a = opGraph.get(match[patternHandles.a]);
  [[maybe_unused]] const auto &b = opGraph.get(match[patternHandles.b]);
  [[maybe_unused]] const auto &out = opGraph.get(match[patternHandles.out]);

  if (m_subgroupSize > m_maxComputeWorkGroupSize[0]) {
    return {};
  }

  std::vector<unsigned int> configs;
  for (unsigned int c = 0; c < CONCAT_CONV_CM_CONFIGS.size(); ++c) {
    const auto &config = CONCAT_CONV_CM_CONFIGS[c];
    uint32_t sgCount = config.wg_n * config.wg_m;
    if (sgCount > m_maxComputeWorkGroupSize[1]) {
      continue;
    }
    uint32_t workgroupInvocations = sgCount * m_subgroupSize;
    if (workgroupInvocations > m_maxComputeWorkGroupInvocations) {
      continue;
    }
    bool a_coopmatShapeSupported = false;
    bool b_coopmatShapeSupported = false;
    for (const auto &shape : m_supportedCoopmatShapes) {
      if (shape.subgroupScope && shape.atype == memory::Dtype::F16 &&
          shape.btype == memory::Dtype::F16 &&
          shape.ctype == memory::Dtype::F16 &&
          shape.acctype == memory::Dtype::F16 && shape.M == config.cm_m &&
          shape.K == config.a_cm_k && shape.N == config.cm_n) {
        a_coopmatShapeSupported = true;
      }
      if (shape.subgroupScope && shape.atype == memory::Dtype::F16 &&
          shape.btype == memory::Dtype::F16 &&
          shape.ctype == memory::Dtype::F16 &&
          shape.acctype == memory::Dtype::F16 && shape.M == config.cm_m &&
          shape.K == config.b_cm_k && shape.N == config.cm_n) {
        b_coopmatShapeSupported = true;
      }
      if (a_coopmatShapeSupported && b_coopmatShapeSupported) {
        break;
      }
    }
    if (!(a_coopmatShapeSupported && b_coopmatShapeSupported)) {
      continue;
    }
    configs.push_back(c);
  }
  return configs;
}

static spirv::GlslCompilerInstance direct_conv_cm_compile(
    spirv::GlslCompiler *compiler, const io::Path &srcPath,
    unsigned int subgroupSize, unsigned int A_C, unsigned int B_C,
    unsigned int K, TensorFormat A_inputFormat, TensorFormat B_inputFormat,
    TensorFormat outputFormat,
    memory::optional<ActivationFunction> activationFunction,
    memory::uvec2 kernelSize, memory::uvec2 padding, memory::uvec2 stride,
    bool bias, const ConcatConvConfig &config,
    //
    memory::FilterLayout *out_A_filterLayout,
    memory::FilterLayout *out_B_filterLayout,
    memory::BiasLayout *out_biasLayout) {
  auto shader = compiler->read(srcPath);
  if (A_C % 8 == 0) {
    shader.define("a_istype", "uvec4");
    shader.define("A_ISTYPE_SIZE", 16);
  } else {
    shader.define("a_istype", "uint16_t");
    shader.define("a_ISTYPE_SIZE", 2);
  }

  if (B_C % 8 == 0) {
    shader.define("b_istype", "uvec4");
    shader.define("B_ISTYPE_SIZE", 16);
  } else {
    shader.define("b_istype", "uint16_t");
    shader.define("B_ISTYPE_SIZE", 2);
  }

  if (K % 8 == 0) {
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
  } else {
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
  }

  if (A_inputFormat == TensorFormat::SSBO_HWC && A_C % 8 == 0) {
    shader.define("A_IN_LAYOUT_HWC8");
  } else if (A_inputFormat == TensorFormat::SSBO_HWC && A_C % 8 != 0) {
    shader.define("A_IN_LAYOUT_HWC");
  } else if (A_inputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("A_IN_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  if (B_inputFormat == TensorFormat::SSBO_HWC && B_C % 8 == 0) {
    shader.define("B_IN_LAYOUT_HWC8");
  } else if (B_inputFormat == TensorFormat::SSBO_HWC && B_C % 8 != 0) {
    shader.define("B_IN_LAYOUT_HWC");
  } else if (B_inputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("B_IN_LAYOUT_CHWC8");
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

  memory::FilterLayout A_filterLayout = memory::FilterLayout::RSCK;
  if ((A_C % config.a_cm_k == 0) &&
      ((config.a_cm_k == 8) || (config.a_cm_k == 16))) {
    if (config.a_cm_k == 8) {
      A_filterLayout = memory::FilterLayout::RSCKC8;
      shader.define("A_FILTER_LAYOUT_RSCKC8");
      shader.define("a_fstype", "uvec4");
      shader.define("A_FSTYPE_SIZE", 16);
    } else if (config.a_cm_k == 16) {
      A_filterLayout = memory::FilterLayout::RSCKC16;
      shader.define("A_FILTER_LAYOUT_RSCKC16");
      shader.define("a_fstype", "uvec4");
      shader.define("A_FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else if (K % config.cm_n == 0 && (config.cm_n == 8 || config.cm_n == 16)) {
    if (config.cm_n == 8) {
      A_filterLayout = memory::FilterLayout::KRSCK8;
      shader.define("A_FILTER_LAYOUT_KRSCK8");
      shader.define("a_fstype", "uvec4");
      shader.define("A_FSTYPE_SIZE", 16);
    } else if (config.cm_n == 16) {
      A_filterLayout = memory::FilterLayout::KRSCK16;
      shader.define("A_FILTER_LAYOUT_KRSCK16");
      shader.define("a_fstype", "uvec4");
      shader.define("A_FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else {
    A_filterLayout = memory::FilterLayout::RSCK;
    shader.define("A_FILTER_LAYOUT_RSCK");
    shader.define("a_fstype", "uint16_t");
    shader.define("A_FSTYPE_SIZE", 2);
  }
  memory::FilterLayout B_filterLayout = memory::FilterLayout::RSCK;
  if ((B_C % config.b_cm_k == 0) &&
      ((config.b_cm_k == 8) || (config.b_cm_k == 16))) {
    if (config.b_cm_k == 8) {
      B_filterLayout = memory::FilterLayout::RSCKC8;
      shader.define("B_FILTER_LAYOUT_RSCKC8");
      shader.define("b_fstype", "uvec4");
      shader.define("B_FSTYPE_SIZE", 16);
    } else if (config.b_cm_k == 16) {
      B_filterLayout = memory::FilterLayout::RSCKC16;
      shader.define("B_FILTER_LAYOUT_RSCKC16");
      shader.define("b_fstype", "uvec4");
      shader.define("B_FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else if (K % config.cm_n == 0 && (config.cm_n == 8 || config.cm_n == 16)) {
    if (config.cm_n == 8) {
      B_filterLayout = memory::FilterLayout::KRSCK8;
      shader.define("B_FILTER_LAYOUT_KRSCK8");
      shader.define("b_fstype", "uvec4");
      shader.define("B_FSTYPE_SIZE", 16);
    } else if (config.cm_n == 16) {
      B_filterLayout = memory::FilterLayout::KRSCK16;
      shader.define("B_FILTER_LAYOUT_KRSCK16");
      shader.define("b_fstype", "uvec4");
      shader.define("B_FSTYPE_SIZE", 16);
    } else {
      diag::invalid_state();
    }
  } else {
    B_filterLayout = memory::FilterLayout::RSCK;
    shader.define("B_FILTER_LAYOUT_RSCK");
    shader.define("b_fstype", "uint16_t");
    shader.define("B_FSTYPE_SIZE", 2);
  }

  if (config.a_async) {
    shader.define("A_ASYNC_READ");
  } else {
    shader.define("A_NASYNC_READ");
  }

  if (config.b_async) {
    shader.define("B_ASYNC_READ");
  } else {
    shader.define("B_NASYNC_READ");
  }

  assert(out_A_filterLayout);
  assert(out_B_filterLayout);
  *out_A_filterLayout = A_filterLayout;
  *out_B_filterLayout = B_filterLayout;

  shader.define("atype", "float16_t");
  shader.define("ATYPE_SIZE", 2);
  shader.define("A_IN_CH", A_C);
  shader.define("B_IN_CH", B_C);
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
  shader.define("A_SG_K", config.a_sg_k);
  shader.define("B_SG_K", config.b_sg_k);
  shader.define("SG_N", config.sg_n);

  shader.define("WG_M", config.wg_m);
  shader.define("WG_N", config.wg_n);

  shader.define("CM_M", config.cm_m);
  shader.define("A_CM_K", config.a_cm_k);
  shader.define("B_CM_K", config.b_cm_k);
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

void ConcatConvCMShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {
  const ConcatConvConfig &config = CONCAT_CONV_CM_CONFIGS[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::EdgeId convId = match[patternHandles.conv];
  memory::NodeId aId = match[patternHandles.a];
  memory::NodeId bId = match[patternHandles.b];
  memory::NodeId outId = match[patternHandles.out];

  const ComputeOp &op = opGraph.get(convId);
  const auto &a = opGraph.get(aId);
  const auto &b = opGraph.get(bId);
  const auto &out = opGraph.get(outId);
  assert(op.tag() == ComputeOpKind::Conv);
  assert(a.channels.isConstant());
  assert(b.channels.isConstant());
  assert(out.channels.isConstant());
  assert(a.width == b.width);
  assert(a.height == b.height);
  assert(a.type == b.type);
  assert(a.width == out.width);
  assert(b.height == out.height);
  const ComputeOpConv &conv = op.conv();

  memory::optional<ActivationFunction> activationFunction;

  if (pattern == CONCAT_CONV_ACTIVATION_PATTERN) {
    activationFunction =
        opGraph.get(match[*m_patternHandles[pattern].relu]).activation().func;
  }

  const uint32_t A_C = static_cast<uint32_t>(a.channels.constant());
  const uint32_t B_C = static_cast<uint32_t>(b.channels.constant());
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());
  const Sym H = a.height;
  const Sym W = a.width;
  const uint32_t R = conv->W->shape().r;
  const uint32_t S = conv->W->shape().s;

  // dummy values; overwritten in direct_conv_cm_compile
  memory::FilterLayout A_filterLayout = memory::FilterLayout::RSCK;
  memory::FilterLayout B_filterLayout = memory::FilterLayout::RSCK;
  memory::BiasLayout biasLayout = memory::BiasLayout::C;

  auto shader = direct_conv_cm_compile(
      m_compiler, m_srcPath, m_subgroupSize, A_C, B_C, K, a.format, b.format,
      out.format, activationFunction,
      memory::uvec2(conv->W->shape().r, conv->W->shape().s), conv->padding,
      conv->stride, conv->B != nullptr, config, //
      &A_filterLayout, &B_filterLayout, &biasLayout);

  std::uint32_t tileX = config.cm_n * config.sg_n * config.wg_n;
  std::uint32_t tileY = config.cm_m;
  std::uint32_t tileZ = config.sg_m * config.wg_m;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX); // 8 / 16 = 1
  Sym workgroupCountY = symGraph.cdiv(W, tileY);            // 1920 / 16 = 120
  Sym workgroupCountZ = symGraph.cdiv(H, tileZ);            // 1080 / 16 = 68

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);

  assert(A_C + B_C == conv->W->shape().c);
  memory::FilterTensor A_filter{
      memory::FilterDescriptor{
          {S, R, A_C, K},
          A_filterLayout,
          memory::Dtype::F16,
      },
  };
  memory::FilterTensor B_filter{
      memory::FilterDescriptor{
          {S, R, B_C, K},
          B_filterLayout,
          memory::Dtype::F16,
      },
  };
  // Split conv->W into two seperate filters, where
  // A_filter contains all input channels < A_C and
  // B_filter contains all input channels >= A_C
  for (uint32_t r = 0; r < R; ++r) {
    for (uint32_t s = 0; s < S; ++s) {
      for (uint32_t c = 0; c < A_C; ++c) {
        for (uint32_t k = 0; k < K; ++k) {
          A_filter.at(s, r, c, k) = conv->W->at(s, r, c, k);
        }
      }
      for (uint32_t c = A_C; c < A_C + B_C; ++c) {
        for (uint32_t k = 0; k < K; ++k) {
          B_filter.at(s, r, c - A_C, k) = conv->W->at(s, r, c, k);
        }
      }
    }
  }

  // TODO: Transform weights lazily!!! (THIS WILL BECOME A BOTTLENECK!!!)
  TensorId A_filterTensor = impl.createParameter(A_filter.desc(), A_filter);
  TensorId B_filterTensor = impl.createParameter(B_filter.desc(), B_filter);

  memory::optional<TensorId> biasTensorId = memory::nullopt;
  if (conv->B != nullptr) {
    biasTensorId = impl.createParameter(
        memory::BiasDescriptor{
            .shape = conv->B->shape(),
            .layout = biasLayout,
            .type = memory::Dtype::F16,
        },
        *conv->B);
  }
  dispatch.addBinding("A_SET", "A_BINDING", Access::ReadOnly, aId);
  dispatch.addBinding("B_SET", "B_BINDING", Access::ReadOnly, bId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addBinding("A_FILTER_SET", "A_FILTER_BINDING", Access::ReadOnly,
                      A_filterTensor);
  dispatch.addBinding("B_FILTER_SET", "B_FILTER_BINDING", Access::ReadOnly,
                      B_filterTensor);
  if (biasTensorId) {
    dispatch.addBinding("BIAS_SET", "BIAS_BINDING", Access::ReadOnly,
                        *biasTensorId);
  }
  dispatch.addPushConstant(PushConstant::Dynamic(W, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(H, memory::Dtype::U32));
  dispatch.setName(name(pattern, configKey));
  dispatch.setSourcePath(m_srcPath);

  Sym inreads = symGraph.mul(symGraph.mul(W, H),
                             (A_C + B_C) * size_of(TensorDataType::Float16));
  size_t wreads = conv->W->byteSize() + (conv->B ? conv->B->byteSize() : 0ull);
  Sym reads = symGraph.add(wreads, inreads);
  Sym writes = symGraph.mul(symGraph.mul(out.width, out.height),
                            K * size_of(TensorDataType::Float16));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);

  dispatch.setDebugInfo(
      fmt::format("{}||{}-direct-conv-{}", a.format, b.format, out.format));
}
memory::string ConcatConvCMShader::name(unsigned int pattern,
                                        unsigned int) const {
  switch (pattern) {
  case CONCAT_CONV_PATTERN:
    return "concat-conv-cm";
  case CONCAT_CONV_ACTIVATION_PATTERN:
    return "concat-conv-cm+activation";
  default:
    diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
