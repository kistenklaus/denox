#include "denox/compiler/implement/shaders/conv/DirectConvShaderCM.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include <fmt/format.h>

namespace denox::compiler::shaders {

DirectConvShaderCM::DirectConvShaderCM(spirv::GlslCompiler *compiler,
                                       const CompileOptions &options)
    : m_compiler(compiler),
      m_enableConvReluFusion(options.features.enableConvReluFusion),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeSharedMemory(
          options.deviceInfo.limits.maxComputeSharedMemory),
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

  // ==== Generate all valid configurations ====
  {
    size_t coopmat_shape_count = 1;
    for (const denox::CoopmatShape &coopmat_shape :
         options.deviceInfo.coopmat.shapes) {
      if (!coopmat_shape.subgroupScope ||
          coopmat_shape.acctype != memory::Dtype::F16 ||
          coopmat_shape.atype != memory::Dtype::F16 ||
          coopmat_shape.btype != memory::Dtype::F16 ||
          coopmat_shape.ctype != memory::Dtype::F16) {
        continue;
      }
      if (coopmat_shape_count-- == 0) {
        break;
      }
      const uint32_t cm_m = coopmat_shape.M;
      const uint32_t cm_k = coopmat_shape.K;
      const uint32_t cm_n = coopmat_shape.N;
      if ((cm_m % 8 != 0) || (cm_k % 8 != 0) || (cm_n % 8 != 0)) {
        continue;
      }

      const uint32_t acc_register_estimate = (cm_m * cm_n) / m_subgroupSize;
      const uint32_t a_register_estimate = (cm_m * cm_k) / m_subgroupSize;
      const uint32_t b_register_estimate = (cm_k * cm_n) / m_subgroupSize;

      // very exhaustive search range!!!
      for (uint32_t wg_m = 1; wg_m < 16; ++wg_m) {
        for (uint32_t wg_n = 1; wg_n < 16; ++wg_n) {
          const uint32_t workgroup_size = wg_m * wg_n * m_subgroupSize;
          if (workgroup_size < 128 ||
              workgroup_size >
                  options.deviceInfo.limits.maxComputeWorkGroupInvocations) {
            continue; // unreasonable workgroup size
          }
          for (uint32_t sg_m = 1; sg_m < 8; ++sg_m) {
            for (uint32_t sg_k = 1; sg_k < 8; ++sg_k) {
              for (uint32_t sg_n = 1; sg_n < 8; ++sg_n) {
                const uint32_t coopmats_register_estimate =
                    acc_register_estimate * sg_n * sg_m +
                    a_register_estimate * sg_m + b_register_estimate * sg_n;
                const uint32_t prefetch_A_QQ = (cm_m * cm_k * sg_k * sg_m) / 8;
                if (prefetch_A_QQ % wg_n != 0) {
                  continue; // uneven load balancing between subgroups.
                }
                const uint32_t prefetch_A_SQQ = prefetch_A_QQ / wg_n;
                // 16bytes word fetched per invocation!
                const uint32_t prefetch_A_IQQ =
                    (prefetch_A_SQQ + m_subgroupSize - 1) / m_subgroupSize;

                const uint32_t prefetch_B_QQ = (cm_k * cm_n * sg_k * sg_n) / 8;
                if (prefetch_B_QQ % wg_m != 0) {
                  continue; // uneven load balancing between subgroups.
                }
                const uint32_t prefetch_B_SQQ = prefetch_B_QQ / wg_m;
                // 16 bytes word fetched per invocation!
                const uint32_t prefetch_B_IQQ =
                    (prefetch_B_SQQ + m_subgroupSize - 1) / m_subgroupSize;

                const uint32_t prefetch_A_register_estimate =
                    prefetch_A_IQQ * 4; // uvec4
                const uint32_t prefetch_B_register_estimate =
                    prefetch_B_IQQ * 4; // uvec4

                const uint32_t register_estimate =
                    coopmats_register_estimate + prefetch_A_register_estimate +
                    prefetch_B_register_estimate;
                if (register_estimate > 160) {
                  continue; // to many registers (conservative limit, because
                            // optimizers might reduce this drastically)
                }
                const uint32_t sh_a_size =
                    (wg_m * cm_m * cm_k * sg_k * sg_m) * 2;
                const uint32_t sh_b_size =
                    (wg_n * cm_k * cm_n * sg_k * sg_n) * 2;
                const uint32_t sh_out_size =
                    wg_m * wg_n * sg_m * sg_n * cm_m * cm_n * 2;
                const uint32_t sh_size =
                    std::max(sh_a_size + sh_b_size, sh_out_size);

                // again very conservative limit especially for this
                // implementation, because it almost doesn't require any shared
                // memory
                static constexpr double WG_SH_OCCUPANCY =
                    0.5; // 75% of max shared memory allowed
                if (static_cast<double>(sh_size) >
                    static_cast<double>(m_maxComputeSharedMemory) *
                        WG_SH_OCCUPANCY) {
                  continue;
                }
                // fmt::println("{}x{}x{}   {}x{}x{}   {}x{}   ->  {}", cm_m,
                // cm_k,
                //              cm_n, sg_m, sg_k, sg_n, wg_m, wg_n, sh_size);

                m_configs.push_back(DirectConvConfigCM{
                    .cm_m = cm_m,
                    .cm_k = cm_k,
                    .cm_n = cm_n,
                    .wg_m = wg_m,
                    .wg_n = wg_n,
                    .sg_m = sg_m,
                    .sg_k = sg_k,
                    .sg_n = sg_n,
                    .async = true,
                });
                m_configs.push_back(DirectConvConfigCM{
                    .cm_m = cm_m,
                    .cm_k = cm_k,
                    .cm_n = cm_n,
                    .wg_m = wg_m,
                    .wg_n = wg_n,
                    .sg_m = sg_m,
                    .sg_k = sg_k,
                    .sg_n = sg_n,
                    .async = false,
                });
              }
            }
          }
        }
      }
    }
  }

  // ==== Define implementable patterns ========
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
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  const auto &conv = opGraph.get(match[patternHandles.conv]).conv();

  if (in.channels.isSymbolic()) {
    return {};
  }
  if (out.channels.isSymbolic()) {
    return {};
  }

  const uint32_t C = static_cast<uint32_t>(in.channels.constant());
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());
  const uint32_t R = conv->W->shape().r;
  const uint32_t S = conv->W->shape().s;

  std::vector<unsigned int> promissing;
  promissing.reserve(m_configs.size());
  for (unsigned int c = 0; c < m_configs.size(); ++c) {
    const auto &config = m_configs[c];

    static constexpr size_t KK_ASYNC_LIMIT = 3;
    static constexpr size_t MAX_CHANNEL_TILE_OVERALLOCATION = 2;
    static constexpr size_t MAX_KTILE_OVERALLOCATION = 2;

    // GEMM loop iterations
    const uint32_t RSC = R * S * C;
    const uint32_t ktile = config.cm_k * config.sg_k;
    const uint32_t KK = (RSC + ktile - 1) / ktile;

    if (KK < KK_ASYNC_LIMIT && config.async) {
      continue; // async doesn't make any sense here!
    }
    if (RSC * MAX_KTILE_OVERALLOCATION < ktile) {
      continue;
    }

    // k over allocation factor.

    uint32_t ctile = config.cm_n * config.sg_n * config.wg_n;
    uint32_t channelDispatchSize = (K + ctile - 1) / ctile;
    if (channelDispatchSize > 1 && K <= 256) {
      // avoid output channel tiling, in cases where implementations
      // without output tiling exist and are most likely a lot better
      continue;
    }

    uint32_t K_eff = std::max(K, config.cm_n);

    if (K_eff * MAX_CHANNEL_TILE_OVERALLOCATION < ctile) {
      continue;
    }

    // POLICY: RSC % ktile == 0
    // NOTE: Only apply if there exist at least one config which achieves cm_k,
    //    which is implied by RSC % config.cm_k == 0, with sg_k = 1
    if (RSC % config.cm_k == 0 && RSC % ktile != 0) {
      continue;
    }

    // POLICY: K % ctile == 0
    if (K % config.sg_n == 0 && K % ctile != 0) {
      continue;
    }

    // POLICY: wgSize \in [128, 256]
    const uint32_t wgSize = config.wg_m * config.wg_n * m_subgroupSize;
    if (wgSize < 128 || wgSize > 256) {
      continue;
    }

    // POLICY: aspect ratio \in [1,2]
    const uint32_t xtile = config.cm_m;
    const uint32_t ytile = config.sg_m * config.wg_m;
    const float aspect = static_cast<float>(xtile) / static_cast<float>(ytile);
    const float eps = 0.001f;
    if (!((aspect + eps) >= 1 && (aspect - eps) <= 2)) {
      // continue;
    }

    // POLICY: tile pressure greater than 32K
    const uint32_t tile_pressue = config.cm_m * config.cm_k * config.cm_n *
                                  config.sg_m * config.sg_k * config.sg_n;
    if (tile_pressue < (1 << 15)) {
      continue;
    }

    // fmt::println("{}x{}x{}   {}x{}x{}   {}x{} -> ({}) {}", config.cm_m,
    // config.cm_k,
    //              config.cm_n, config.sg_m, config.sg_k, config.sg_n,
    //              config.wg_m, config.wg_n, tile_pressue, tile_pressue > (1 <<
    //              15));

    promissing.push_back(c);
  }
  fmt::println("config count = {}", promissing.size());
  return promissing;
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
  const DirectConvConfigCM &config = m_configs[configKey];

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

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(in.width, tileY);
  Sym workgroupCountZ = symGraph.cdiv(in.height, tileZ);

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
  dispatch.addParamBinding("FILTER_SET", "FILTER_BINDING", weightTensorId);
  if (biasTensorId) {
    dispatch.addParamBinding("BIAS_SET", "BIAS_BINDING", *biasTensorId);
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

  Sym flops =
      symGraph.mul(symGraph.mul(out.width, out.height),
                   2ull * C * K * conv->W->shape().r * conv->W->shape().s);
  dispatch.setFlops(flops);

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
  dispatch.usesCoopmat(true);
  dispatch.setName(name());
  dispatch.setSourcePath(m_srcPath);

  dispatch.setConfig(fmt::format("CM_M={}#CM_K={}#CM_N={}#SG_M={}#SG_K={}#SG_N="
                                 "{}#WG_M={}#WG_N={}#ASYNC={}",
                                 config.cm_m, config.cm_k, config.cm_n,
                                 config.sg_m, config.sg_k, config.sg_n,
                                 config.wg_m, config.wg_n, config.async));
}

memory::string DirectConvShaderCM::name() const { return "direct-conv-cm"; }
} // namespace denox::compiler::shaders
