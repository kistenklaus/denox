#include "denox/compiler/implement/shaders/conv/DirectConvShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include <cassert>
#include <flatbuffers/base.h>
#include <fmt/format.h>

namespace denox::compiler::shaders {

struct DirectConvConfig {
  unsigned int invoc_m;
  unsigned int invoc_k;
  unsigned int invoc_n;
  unsigned int wg_m;
  unsigned int wg_n;
  unsigned int sg_m;
  unsigned int sg_k;
  unsigned int sg_n;
  bool async;
};

static constexpr DirectConvConfig DIRECT_CONV_CONFIGS[] = {
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 1,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    DirectConvConfig{
        // <- good for 32 -> 32 (but equal to 2,2,2)
        .invoc_m = 16,
        .invoc_k = 16,
        .invoc_n = 16,
        .wg_m = 4,
        .wg_n = 2,
        .sg_m = 2,
        .sg_k = 2,
        .sg_n = 4,
        .async = true,
    },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 4,
    //     .wg_n = 2,
    //     .sg_m = 2,
    //     .sg_k = 2,
    //     .sg_n = 2,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 4,
    //     .wg_n = 2,
    //     .sg_m = 2,
    //     .sg_k = 1,
    //     .sg_n = 8,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 6,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 7,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 1,
    //     .sg_n = 8,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 6,
    //     .async = false,
    // },
    // DirectConvConfig{
    //     .invoc_m = 16,
    //     .invoc_k = 16,
    //     .invoc_n = 16,
    //     .wg_m = 8,
    //     .wg_n = 1,
    //     .sg_m = 1,
    //     .sg_k = 3,
    //     .sg_n = 7,
    //     .async = false,
    // },
};

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
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);

  // TODO: Remove this restriction!

  if (m_subgroupSize > m_maxComputeWorkGroupSize[0]) {
    fmt::println("invalid subgroup size");
    return {};
  }

  if (in.channels.isSymbolic()) {
    fmt::println("no config for symbolic channel count");
    return {};
  }

  if (out.channels.isSymbolic()) {
    fmt::println("no config for symbolic channel count");
    return {};
  }
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());

  std::vector<unsigned int> configs;
  for (unsigned int i = 0;
       i < sizeof(DIRECT_CONV_CONFIGS) / sizeof(DirectConvConfig); ++i) {
    const auto &config = DIRECT_CONV_CONFIGS[i];
    std::uint32_t channelTile = config.invoc_n * config.sg_n * config.wg_n;
    assert(config.invoc_n % 8 == 0 && "INVOC_N must be multiple of 8");
    assert(config.invoc_k % 8 == 0 && "INVOC_K must be multiple of 8");
    assert((config.invoc_n * config.sg_n * config.wg_n) % 8 == 0 &&
           "WG_TILE_N must be multiple of 8");
    if (K > channelTile) {
      continue;
    }
    configs.push_back(i);
  }
  if (configs.empty()) {
    fmt::println("no config for K={}", K);
  }

  return configs;
}

spirv::GlslCompilerInstance
direct_conv_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                    unsigned int subgroupSize, unsigned int C, unsigned int K,
                    TensorFormat inputFormat, TensorFormat outputFormat,
                    memory::optional<ActivationFunction> activationFunction,
                    memory::uvec2 kernelSize, memory::uvec2 padding,
                    memory::uvec2 stride, bool bias,
                    const DirectConvConfig &config,
                    memory::FilterLayout *out_filterLayout,
                    memory::BiasLayout *out_biasLayout) {
  assert(config.invoc_n % 8 == 0 && "INVOC_N must be multiple of 8");
  assert(config.invoc_k % 8 == 0 && "INVOC_K must be multiple of 8");

  assert((config.invoc_n * config.sg_n * config.wg_n) % 8 == 0 &&
         "WG_TILE_N must be multiple of 8");

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
  *out_filterLayout = memory::FilterLayout::RSc8k8;
  shader.define("FILTER_LAYOUT_RSC8K8");
  shader.define("fstype", "uvec4");
  shader.define("FSTYPE_SIZE", 16);

  // TODO:
  assert(config.invoc_n * config.sg_n * config.wg_n >= K &&
         "direct_conv expects a KRc8k<invoc_c*sg_n*wg_n cdiv 8> layout, this "
         "is identical to RSc8k8, unless  "
         "the output channels are tiled. If this one fails we should simply "
         "implement the actual layout needed as a parameterized layout, a bit "
         "more difficult, but that should do it.");

  if (config.async) {
    shader.define("ASYNC_READ");
  } else {
    shader.define("NASYNC_READ");
  }

  shader.define("atype", "float16_t");
  shader.define("ATYPE_SIZE", 2);
  shader.define("IN_CH", C);
  shader.define("OUT_CH", K);

  shader.define("SG_SIZE", subgroupSize);

  shader.define("SG_M", config.sg_m);
  shader.define("SG_K", config.sg_k);
  shader.define("SG_N", config.sg_n);

  shader.define("WG_M", config.wg_m);
  shader.define("WG_N", config.wg_n);

  shader.define("INVOC_M", config.invoc_m);
  shader.define("INVOC_K", config.invoc_k);
  shader.define("INVOC_N", config.invoc_n);

  unsigned int subgroupCount = config.wg_m * config.wg_n;
  shader.define("SG_COUNT", subgroupCount);

  shader.define("KERNEL_X", kernelSize.x);
  shader.define("KERNEL_Y", kernelSize.y);
  shader.define("STRIDE_X", stride.x);
  shader.define("STRIDE_Y", stride.y);
  shader.define("PADDING_X", padding.x);
  shader.define("PADDING_Y", padding.y);

  if (bias) {
    shader.define("USE_BIAS");
    if (config.invoc_n == 8) {
      *out_biasLayout = memory::BiasLayout::C8;
    } else if (config.invoc_n == 16) {
      *out_biasLayout = memory::BiasLayout::C16;
    } else {
      *out_biasLayout = memory::BiasLayout::C;
    }
  } else {
    shader.define("NUSE_BIAS");
  }
  return shader;
}

void DirectConvShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {

  const DirectConvConfig &config = DIRECT_CONV_CONFIGS[configKey];

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

  auto shader = direct_conv_compile(
      m_compiler, m_srcPath, m_subgroupSize, C, K, in.format, out.format,
      activationFunction, memory::uvec2(conv->W->shape().r, conv->W->shape().s),
      conv->padding, conv->stride, conv->B != nullptr, config, &filterLayout,
      &biasLayout);
  // fmt::println("PREAMBLE:\n{}", shader.getPreamble());

  std::uint32_t tileX = config.invoc_n * config.sg_n * config.wg_n;
  std::uint32_t tileY = config.invoc_m;
  std::uint32_t tileZ = config.sg_m * config.wg_m;

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

  Sym inreads =
      symGraph.mul(symGraph.mul(in.width, in.height), C * size_of(in.type));
  size_t wreads = conv->W->byteSize() + (conv->B ? conv->B->byteSize() : 0ull);
  Sym reads = symGraph.add(wreads, inreads);
  Sym writes =
      symGraph.mul(symGraph.mul(out.width, out.height), K * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);

  dispatch.setFlops(
      symGraph.mul(symGraph.mul(out.width, out.height),
                   2 * C * K * conv->W->shape().r * conv->W->shape().s));

  if (activationFunction) {
    switch (*activationFunction) {
    case ActivationFunction::ReLU:
      dispatch.setOperation(fmt::format(
          "relu(conv2d(x,kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)))",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
      break;
    case ActivationFunction::LeakyReLU:
      dispatch.setOperation(fmt::format(
          "leaky_relu(conv2d(x,kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)))",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
      break;
    case ActivationFunction::SiLU:
      dispatch.setOperation(fmt::format(
          "silu(conv2d(x,kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)))",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
      break;
    }
  } else {
    dispatch.setOperation(fmt::format(
        "conv2d(x,kernel_size=({},{}),bias={},stride=({},"
        "{}),padding=({},{}),dialation=(1,1))",
        conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
        conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
  }
  dispatch.usesCoopmat(false);
  dispatch.setName(name());
  dispatch.setSourcePath(m_srcPath);
  dispatch.setConfig(fmt::format("INVOC_M={}#INVOC_K={}#INVOC_N={}#SG_M={}#SG_"
                                 "K={}#SG_N={}#WG_M={}#WG_N={}#ASYNC={}",
                                 config.invoc_m, config.invoc_k, config.invoc_n,
                                 config.sg_m, config.sg_k, config.sg_n,
                                 config.wg_m, config.wg_n, config.async));
}
memory::string DirectConvShader::name() const { return "DirectConvShader"; }
} // namespace denox::compiler::shaders
