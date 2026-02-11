#include "denox/compiler/implement/shaders/conv/ConcatConvCMShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FitlerDescriptor.hpp"
#include <fmt/format.h>

namespace denox::compiler::shaders {

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
  if (!options.features.enableConcatConvFusion) {
    return;
  }
  if (!options.features.coopmat) {
    return;
  }
  if (options.deviceInfo.coopmat.supported == false) {
    return;
  }
  { // Generate config space.

    memory::small_vector<std::pair<uint32_t, denox::CoopmatShape>, 3>
        coopmatShapes;
    static constexpr size_t COOPMAT_SHAPE_SPACE = 1;
    for (const denox::CoopmatShape &shape : options.deviceInfo.coopmat.shapes) {
      if (!shape.subgroupScope || shape.acctype != memory::Dtype::F16 ||
          shape.atype != memory::Dtype::F16 ||
          shape.btype != memory::Dtype::F16 ||
          shape.ctype != memory::Dtype::F16) {
        continue;
      }
      if ((shape.M % 8 != 0) || (shape.K % 8 != 0) || (shape.N % 8 != 0)) {
        continue;
      }
      if (shape.M == 16 && shape.K == 16 && shape.N == 16) {
        coopmatShapes.emplace_back(10, shape);
      } else if (shape.M == 16 && shape.K == 8 && shape.N == 8) {
        coopmatShapes.emplace_back(5, shape);
      } else if (shape.M == 16 && shape.K == 8 && shape.N == 16) {
        coopmatShapes.emplace_back(8, shape);
      } else {
        coopmatShapes.emplace_back(0, shape);
      }
    }
    std::ranges::sort(coopmatShapes, [](const auto &lhs, const auto &rhs) {
      return lhs.first >= rhs.first;
    });
    coopmatShapes.resize(
        std::min<size_t>(coopmatShapes.size(), COOPMAT_SHAPE_SPACE));
    for (const auto &[_, cm_a] : coopmatShapes) {
      const uint32_t cm_m = cm_a.M;
      const uint32_t a_cm_k = cm_a.K;
      const uint32_t cm_n = cm_a.N;

      for (const auto &[_, cm_b] : coopmatShapes) {
        if (cm_a.M != cm_b.M) {
          continue;
        }
        if (cm_a.N != cm_b.N) {
          continue;
        }
        const uint32_t b_cm_k = cm_b.K;

        const uint32_t acc_register_estimate = (cm_m * cm_n) / m_subgroupSize;

        const uint32_t A_a_register_estimate = (cm_m * a_cm_k) / m_subgroupSize;
        const uint32_t A_b_register_estimate = (a_cm_k * cm_n) / m_subgroupSize;

        const uint32_t B_a_register_estimate = (cm_m * b_cm_k) / m_subgroupSize;
        const uint32_t B_b_register_estimate = (b_cm_k * cm_n) / m_subgroupSize;

        for (uint32_t wg_m = 1; wg_m < 16; ++wg_m) {
          for (uint32_t wg_n = 1; wg_n < 16; ++wg_n) {
            const uint32_t workgroup_size = wg_m * wg_n * m_subgroupSize;
            if (workgroup_size < 128 ||
                workgroup_size >
                    options.deviceInfo.limits.maxComputeWorkGroupInvocations) {
              continue; // unreasonable workgroup size
            }
            for (uint32_t sg_m = 1; sg_m < 8; ++sg_m) {
              for (uint32_t sg_n = 1; sg_n < 8; ++sg_n) {
                for (uint32_t a_sg_k = 1; a_sg_k < 8; ++a_sg_k) {
                  for (uint32_t b_sg_k = 1; b_sg_k < 8; ++b_sg_k) {
                    const uint32_t coopmats_register_estimate =
                        acc_register_estimate * sg_n * sg_m +
                        std::min(A_a_register_estimate * sg_m +
                                     A_b_register_estimate * sg_n,
                                 B_a_register_estimate * sg_m *
                                     B_b_register_estimate * sg_n);

                    const uint32_t A_prefetch_A_QQ =
                        (cm_m * a_cm_k * a_sg_k * sg_m) / 8;

                    const uint32_t B_prefetch_A_QQ =
                        (cm_m * b_cm_k * b_sg_k * sg_m) / 8;

                    const uint32_t A_prefetch_A_SQQ = A_prefetch_A_QQ / wg_n;
                    const uint32_t B_prefetch_A_SQQ = B_prefetch_A_QQ / wg_n;
                    // 16bytes word fetched per invocation!
                    const uint32_t A_prefetch_A_IQQ =
                        (A_prefetch_A_SQQ + m_subgroupSize - 1) /
                        m_subgroupSize;
                    const uint32_t B_prefetch_A_IQQ =
                        (B_prefetch_A_SQQ + m_subgroupSize - 1) /
                        m_subgroupSize;

                    const uint32_t A_prefetch_B_QQ =
                        (a_cm_k * cm_n * a_sg_k * sg_n) / 8;
                    if (A_prefetch_B_QQ % wg_m != 0) {
                      continue; // uneven load balancing between subgroups.
                    }

                    const uint32_t B_prefetch_B_QQ =
                        (b_cm_k * cm_n * b_sg_k * sg_n) / 8;
                    if (B_prefetch_B_QQ % wg_m != 0) {
                      continue; // uneven load balancing between subgroups.
                    }

                    const uint32_t A_prefetch_B_SQQ = A_prefetch_B_QQ / wg_m;
                    const uint32_t B_prefetch_B_SQQ = B_prefetch_B_QQ / wg_m;

                    const uint32_t A_prefetch_B_IQQ =
                        (A_prefetch_B_SQQ + m_subgroupSize - 1) /
                        m_subgroupSize;

                    const uint32_t B_prefetch_B_IQQ =
                        (B_prefetch_B_SQQ + m_subgroupSize - 1) /
                        m_subgroupSize;

                    const uint32_t A_prefetch_A_register_estimate =
                        A_prefetch_A_IQQ * 4; // uvec4
                    const uint32_t A_prefetch_B_register_estimate =
                        A_prefetch_B_IQQ * 4; // uvec4

                    const uint32_t B_prefetch_A_register_estimate =
                        B_prefetch_A_IQQ * 4; // uvec4
                    const uint32_t B_prefetch_B_register_estimate =
                        B_prefetch_B_IQQ * 4; // uvec4

                    const uint32_t A_prefetch_register_estimate =
                        A_prefetch_A_register_estimate +
                        A_prefetch_B_register_estimate;

                    const uint32_t B_prefetch_register_estimate =
                        B_prefetch_A_register_estimate +
                        B_prefetch_B_register_estimate;

                    const uint32_t register_estimate =
                        coopmats_register_estimate +
                        std::max(A_prefetch_register_estimate,
                                 B_prefetch_register_estimate);

                    // register estimate is only proportional to the register
                    // counts, so there is a good chance that 160 estimate
                    // corresponds to only 50-60 live registers at a time.
                    if (register_estimate > 160) {
                      continue; // to many registers (conservative limit,
                                // because optimizers might reduce this
                                // drastically)
                    }

                    const uint32_t A_sh_a_size =
                        (wg_m * cm_m * a_cm_k * a_sg_k * sg_m) * 2;

                    const uint32_t B_sh_a_size =
                        (wg_m * cm_m * b_cm_k * b_sg_k * sg_m) * 2;

                    const uint32_t A_sh_b_size =
                        (wg_n * a_cm_k * cm_n * a_sg_k * sg_n) * 2;

                    const uint32_t B_sh_b_size =
                        (wg_n * b_cm_k * cm_n * b_sg_k * sg_n) * 2;

                    const uint32_t sh_out_size =
                        wg_m * wg_n * sg_m * sg_n * cm_m * cm_n * 2;

                    const uint32_t A_sh_size =
                        std::max(A_sh_a_size + A_sh_b_size, sh_out_size);

                    const uint32_t B_sh_size =
                        std::max(B_sh_a_size + B_sh_b_size, sh_out_size);

                    const uint32_t sh_size = std::max(A_sh_size, B_sh_size);

                    static constexpr double WG_SH_OCCUPANCY =
                        0.5; // 75% of max shared memory allowed
                    if (static_cast<double>(sh_size) >
                        static_cast<double>(
                            options.deviceInfo.limits.maxComputeSharedMemory) *
                            WG_SH_OCCUPANCY) {
                      continue;
                    }

                    m_configs.push_back(ConcatConvConfig{
                        .cm_m = cm_m,
                        .a_cm_k = a_cm_k,
                        .b_cm_k = b_cm_k,
                        .cm_n = cm_n,
                        .wg_m = wg_m,
                        .wg_n = wg_n,
                        .sg_m = sg_m,
                        .a_sg_k = a_sg_k,
                        .b_sg_k = b_sg_k,
                        .sg_n = sg_n,
                        .a_async = false,
                        .b_async = false,
                    });

                    // m_configs.push_back(ConcatConvConfig{
                    //     .cm_m = cm_m,
                    //     .a_cm_k = a_cm_k,
                    //     .b_cm_k = b_cm_k,
                    //     .cm_n = cm_n,
                    //     .wg_m = wg_m,
                    //     .wg_n = wg_n,
                    //     .sg_m = sg_m,
                    //     .a_sg_k = a_sg_k,
                    //     .b_sg_k = b_sg_k,
                    //     .sg_n = sg_n,
                    //     .a_async = true,
                    //     .b_async = false,
                    // });
                    // m_configs.push_back(ConcatConvConfig{
                    //     .cm_m = cm_m,
                    //     .a_cm_k = a_cm_k,
                    //     .b_cm_k = b_cm_k,
                    //     .cm_n = cm_n,
                    //     .wg_m = wg_m,
                    //     .wg_n = wg_n,
                    //     .sg_m = sg_m,
                    //     .a_sg_k = a_sg_k,
                    //     .b_sg_k = b_sg_k,
                    //     .sg_n = sg_n,
                    //     .a_async = false,
                    //     .b_async = true,
                    // });
                    m_configs.push_back(ConcatConvConfig{
                        .cm_m = cm_m,
                        .a_cm_k = a_cm_k,
                        .b_cm_k = b_cm_k,
                        .cm_n = cm_n,
                        .wg_m = wg_m,
                        .wg_n = wg_n,
                        .sg_m = sg_m,
                        .a_sg_k = a_sg_k,
                        .b_sg_k = b_sg_k,
                        .sg_n = sg_n,
                        .a_async = true,
                        .b_async = true,
                    });

                    // fmt::println("{}x{}/{}x{}   {}x{}/{}x{}   {}x{}   -> {}",
                    //              cm_m, a_cm_k, b_cm_k, cm_n, sg_m, a_sg_k,
                    //              b_sg_k, sg_n, wg_m, wg_n, sh_size);
                  }
                }
              }
            }
          }
        }
      }
    }
    // fmt::println("config space: {}", m_configs.size());
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
    relu->matchValue([](const ComputeOp &op) -> bool {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return (func.kind() == ActivationFunctionKind::ReLU) ||
             (func.kind() == ActivationFunctionKind::LeakyReLU);
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
  [[maybe_unused]] const auto &conv =
      opGraph.get(match[patternHandles.conv]).conv();
  [[maybe_unused]] const auto &out = opGraph.get(match[patternHandles.out]);

  if (m_subgroupSize > m_maxComputeWorkGroupSize[0]) {
    return {};
  }

  const uint32_t A_C = static_cast<uint32_t>(a.channels.constant());
  const uint32_t B_C = static_cast<uint32_t>(b.channels.constant());
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());
  const uint32_t R = conv->W->shape().r;
  const uint32_t S = conv->W->shape().s;

  // if (A_C == 64 && B_C == 32 && K == 64) {
  //   if (a.format != TensorFormat::SSBO_HWC) {
  //     return {};
  //   }
  //   if (b.format != TensorFormat::SSBO_HWC) {
  //     return {};
  //   }
  //   if (out.format != TensorFormat::SSBO_HWC) {
  //     return {};
  //   }
  //   auto it = std::ranges::find_if(
  //       m_configs, [](const ConcatConvConfig &config) -> bool {
  //         return config.cm_m == 16 && config.a_cm_k == 16 &&
  //                config.b_cm_k == 16 //
  //                && config.cm_n == 16 && config.sg_m == 4 &&
  //                config.a_sg_k == 3 && config.b_sg_k == 3 && config.sg_n == 2
  //                //
  //                && config.wg_m == 2 && config.wg_n == 2 //
  //                && config.a_async == true && config.b_async == true;
  //       });
  //
  //   assert(it != m_configs.end());
  //   uint32_t c = static_cast<uint32_t>(std::distance(m_configs.begin(), it));
  //   return {c};
  // }

  memory::vector<unsigned int> promissing;
  for (uint32_t c = 0; c < m_configs.size(); ++c) {

    static constexpr size_t KK_ASYNC_LIMIT = 3;
    static constexpr size_t MAX_CHANNEL_TILE_OVERALLOCATION = 2;
    static constexpr size_t MAX_KTILE_OVERALLOCATION = 2;
    const auto &config = m_configs[c];
    const uint32_t A_RSC = R * S * A_C;
    const uint32_t B_RSC = R * S * B_C;

    const uint32_t A_ktile = config.a_cm_k * config.a_sg_k;
    const uint32_t B_ktile = config.b_cm_k * config.b_sg_k;

    const uint32_t A_KK = (A_RSC + A_ktile - 1) / A_ktile;
    const uint32_t B_KK = (B_RSC + B_ktile - 1) / B_ktile;

    if (A_KK <= 2 && !config.a_async) {
      continue;
    }
    if (B_KK <= 2 && !config.b_async) {
      continue;
    }
    if (A_KK >= 9 && !config.a_async) {
      continue;
    }
    if (B_KK >= 9 && !config.b_async) {
      continue;
    }

    // fmt::println("{}x{}/{}x{}   {}x{}/{}x{}   {}x{}", config.cm_m,
    // config.a_cm_k,
    //              config.b_cm_k, config.cm_n, config.sg_m, config.a_sg_k,
    //              config.b_sg_k, config.sg_n, config.wg_m, config.wg_n);

    // output channel tile!
    const uint32_t ctile = config.cm_n * config.sg_n * config.wg_n;
    uint32_t channelDispatchSize = (K + ctile - 1) / ctile;
    if (channelDispatchSize > 1 && K <= 256) {
      continue;
    }

    uint32_t K_eff = std::max(K, config.cm_n);
    if (K_eff * MAX_CHANNEL_TILE_OVERALLOCATION < ctile) {
      continue;
    }

    // POLICY: K % ctile == 0
    if (K % config.cm_n == 0) {
      if (K % ctile != 0) {
        continue;
      }
    } else {
      uint32_t wasted = ctile - (K % ctile);
      if (wasted > config.cm_n) {
        // NOTE: kind of assumptious
        // might run into cases where we generate no configs,
        // or only really shity sg_n = 1 configs
        continue;
      }
    }

    if (A_RSC * MAX_KTILE_OVERALLOCATION < A_ktile) {
      continue;
    }
    if (B_RSC * MAX_KTILE_OVERALLOCATION < B_ktile) {
      continue;
    }

    if (A_RSC % config.a_cm_k == 0) {
      if (A_RSC % A_ktile != 0) {
        continue;
      }
    } else {
      const uint32_t wasted = A_ktile - (A_RSC % A_ktile);
      if (wasted > config.a_cm_k) {
        continue;
      }
    }
    if (B_RSC % config.b_cm_k == 0) {
      if (B_RSC % B_ktile != 0) {
        continue;
      }
    } else {
      const uint32_t wasted = B_ktile - (B_RSC % B_ktile);
      if (wasted > config.b_cm_k) {
        continue;
      }
    }

    // POLICY: wgSize \in [128, 256]
    const uint32_t wgSize = config.wg_m * config.wg_n * m_subgroupSize;
    if (wgSize < 128 || wgSize > 256) {
      continue;
    }
    promissing.push_back(c);
  }
  // fmt::println("concat-conv-promissing: {}", promissing.size());
  return promissing;
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
    diag::invalid_state("ConcatConvCMShader: Invalid A_inputFormat, during "
                        "GLSL macro selection.");
  }

  if (B_inputFormat == TensorFormat::SSBO_HWC && B_C % 8 == 0) {
    shader.define("B_IN_LAYOUT_HWC8");
  } else if (B_inputFormat == TensorFormat::SSBO_HWC && B_C % 8 != 0) {
    shader.define("B_IN_LAYOUT_HWC");
  } else if (B_inputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("B_IN_LAYOUT_CHWC8");
  } else {
    diag::invalid_state("ConcatConvCMShader: Invalid B_inputFormat, during "
                        "GLSL macro selection.");
  }

  if (outputFormat == TensorFormat::SSBO_HWC && K % 8 == 0) {
    shader.define("OUT_LAYOUT_HWC8");
  } else if (outputFormat == TensorFormat::SSBO_HWC && K % 8 != 0) {
    shader.define("OUT_LAYOUT_HWC");
  } else if (outputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    diag::invalid_state("ConcatConvCMShader: Invalid outputFormat, during GLSL "
                        "macro selection.");
  }

  if (activationFunction) {
    switch (activationFunction->kind()) {
    case ActivationFunctionKind::ReLU:
      shader.define("ACTIVATION_ReLU");
      break;
    case ActivationFunctionKind::LeakyReLU:
      shader.define("ACTIVATION_LeakyReLU");
      shader.define(
          "ACTIVATION_LeakyReLU_alpha",
          fmt::format("({}f)", activationFunction->leaky_relu().alpha));
      break;
    case ActivationFunctionKind::SiLU:
    case denox::ActivationFunctionKind::Swish:
      diag::invalid_state("ConcatConvCMShader: Invalid activationFunction, "
                          "during GLSL mascro selection.");
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
      diag::invalid_state(
          "ConcatConvCMShader: Invalid cooperative matrix shape, during "
          "GLSL macro selection, of A_filterLayout.");
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
      diag::invalid_state(
          "ConcatConvCMShader: Invalid cooperative matrix shape, during "
          "GLSL macro selection, of A_filterLayout.");
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
      diag::invalid_state(
          "ConcatConvCMShader: Invalid cooperative matrix shape, during "
          "GLSL macro selection, of B_filterLayout.");
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
      diag::invalid_state(
          "ConcatConvCMShader: Invalid cooperative matrix shape, during "
          "GLSL macro selection, of B_filterLayout.");
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

  const ConcatConvConfig &config = m_configs[configKey];

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

  Sym workgroupCountX =
      symGraph.cdiv(out.channels, tileX, false, false); // 8 / 16 = 1
  Sym workgroupCountY =
      symGraph.cdiv(W, tileY, false, false); // 1920 / 16 = 120
  Sym workgroupCountZ = symGraph.cdiv(H, tileZ, false, false); // 1080 / 16 = 68

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);

  assert(A_C + B_C == conv->W->shape().c);

  memory::FilterDescriptor A_filterDesc{
      {S, R, A_C, K},
      A_filterLayout,
      memory::Dtype::F16,
  };
  TensorId A_filterTensor = impl.createParameter(
      A_filterDesc.byteSize(), TensorDataType::Float16,
      TensorStorage::StorageBuffer, TensorFormat::Optimal,
      [W = conv->W, desc = A_filterDesc]() -> std::vector<std::byte> {
        memory::FilterTensor filter{desc};
        for (uint32_t r = 0; r < desc.shape.r; ++r) {
          for (uint32_t s = 0; s < desc.shape.s; ++s) {
            for (uint32_t c = 0; c < desc.shape.c; ++c) {
              for (uint32_t k = 0; k < desc.shape.k; ++k) {
                filter.at(s, r, c, k) = W->at(s, r, c, k);
              }
            }
          }
        }
        std::vector<std::byte> raw(filter.span().begin(), filter.span().end());
        return raw;
      });

  memory::FilterDescriptor B_filterDesc{
      {S, R, B_C, K}, B_filterLayout, memory::Dtype::F16};
  TensorId B_filterTensor = impl.createParameter(
      B_filterDesc.byteSize(), TensorDataType::Float16,
      TensorStorage::StorageBuffer, TensorFormat::Optimal,
      [W = conv->W, desc = B_filterDesc, A_C]() -> std::vector<std::byte> {
        memory::FilterTensor filter{desc};
        for (uint32_t r = 0; r < desc.shape.r; ++r) {
          for (uint32_t s = 0; s < desc.shape.s; ++s) {
            for (uint32_t c = 0; c < desc.shape.c; ++c) {
              for (uint32_t k = 0; k < desc.shape.k; ++k) {
                filter.at(s, r, c, k) = W->at(s, r, c + A_C, k);
              }
            }
          }
        }
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

  dispatch.addBinding("A_SET", "A_BINDING", Access::ReadOnly, aId);
  dispatch.addBinding("B_SET", "B_BINDING", Access::ReadOnly, bId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addParamBinding("A_FILTER_SET", "A_FILTER_BINDING", A_filterTensor);
  dispatch.addParamBinding("B_FILTER_SET", "B_FILTER_BINDING", B_filterTensor);
  if (biasTensorId) {
    dispatch.addParamBinding("BIAS_SET", "BIAS_BINDING", *biasTensorId);
  }
  dispatch.addPushConstant(PushConstant::Dynamic(W, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(H, memory::Dtype::U32));
  dispatch.setSourcePath(m_srcPath);

  Sym inreads = symGraph.mul(symGraph.mul(W, H),
                             (A_C + B_C) * size_of(TensorDataType::Float16));
  size_t wreads = conv->W->byteSize() + (conv->B ? conv->B->byteSize() : 0ull);
  Sym reads = symGraph.add(wreads, inreads);
  Sym writes = symGraph.mul(symGraph.mul(out.width, out.height),
                            K * size_of(TensorDataType::Float16));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.usesCoopmat(true);
  dispatch.setFlops(symGraph.mul(symGraph.mul(out.width, out.height),
                                 2 * (A_C + B_C) * K * conv->W->shape().r *
                                     conv->W->shape().s));

  if (activationFunction) {
    switch (activationFunction->kind()) {
    case ActivationFunctionKind::ReLU:
      dispatch.setOperation(fmt::format(
          "relu(conv2d(concat([x,y],0),kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)))",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y,

          conv->W->shape().s, conv->W->shape().r));
      break;
    case ActivationFunctionKind::LeakyReLU:
      dispatch.setOperation(fmt::format(
          "leaky_relu(conv2d(concat([x,y],0),kernel_size=({},{}),bias={},"
          "stride=({},"
          "{}),padding=({},{}),dialation=(1,1)),alpha={})",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y,
          activationFunction->leaky_relu().alpha));
      break;
    case ActivationFunctionKind::SiLU:
      dispatch.setOperation(fmt::format(
          "silu(conv2d(concat([x,y],0),kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)))",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
      break;
    case ActivationFunctionKind::Swish:
      dispatch.setOperation(fmt::format(
          "swish(conv2d(concat([x,y],0),kernel_size=({},{}),bias={},stride=({},"
          "{}),padding=({},{}),dialation=(1,1)),beta={})",
          conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
          conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y,
          activationFunction->swish().beta));
      break;
    }
  } else {
    dispatch.setOperation(fmt::format(
        "conv2d(concat([x,y],0),kernel_size=({},{}),bias={},stride=({},"
        "{}),padding=({},{}),dialation=(1,1))",
        conv->W->shape().s, conv->W->shape().r, conv->B != nullptr,
        conv->stride.x, conv->stride.y, conv->padding.x, conv->padding.y));
  }
  dispatch.setConfig(
      fmt::format("CM_M={}#A_CM_K={}#B_CM_K={}#CM_N={}#SG_M={}#A_SG_K={}#B_SG_"
                  "K={}#SG_N={}#WG_M={}#WG_N={}#A_ASYNC={}#B_ASYNC={}",
                  config.cm_m, config.a_cm_k, config.b_cm_k, config.cm_n,
                  config.sg_m, config.a_sg_k, config.b_sg_k, config.sg_n,
                  config.wg_m, config.wg_n, config.a_async, config.b_async));

  dispatch.setName(name());
}
memory::string ConcatConvCMShader::name() const { return "concat-conv-cm"; }
} // namespace denox::compiler::shaders
