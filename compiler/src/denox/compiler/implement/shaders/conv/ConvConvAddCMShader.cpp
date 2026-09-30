#include "denox/compiler/implement/shaders/conv/ConvConvAddCMShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/ComputeOp.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include <algorithm>
#include <fmt/format.h>

namespace denox::compiler::shaders {

ConvConvAddCMShader::ConvConvAddCMShader(spirv::GlslCompiler *compiler,
                                         const CompileOptions &options)
    : m_compiler(compiler),
      m_subgroupControl(
          options.deviceInfo.subgroup.controlProperties.supported &&
          options.deviceInfo.subgroup.controlProperties.supportedSubgroupSizes
                  .size() > 1),
      m_optimizationLevel(options.optimizationLevel) {

  if (options.deviceInfo.subgroup.subgroupSize == 0) {
    return;
  }
  if (!options.deviceInfo.subgroup.supportsBasicOps) {
    return;
  }
  if (!options.deviceInfo.subgroup.supportsBallotOps) {
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

  if (options.optimizationLevel >= 5) {
    { // Generate config space.

      memory::small_vector<uint32_t, 2> subgroupSizes;
      if (options.optimizationLevel >= 5 &&
          options.deviceInfo.subgroup.controlProperties.supported) {
        subgroupSizes = options.deviceInfo.subgroup.controlProperties
                            .supportedSubgroupSizes;
      } else {
        if (options.deviceInfo.subgroup.controlProperties.supported &&
            std::ranges::count(options.deviceInfo.subgroup.controlProperties
                                   .supportedSubgroupSizes,
                               32) != 0) {
          subgroupSizes = {32};
        } else {
          subgroupSizes.push_back(options.deviceInfo.subgroup.subgroupSize);
        }
      };

      for (const uint32_t subgroupSize : subgroupSizes) {

        memory::small_vector<std::pair<uint32_t, denox::CoopmatShape>, 3>
            coopmatShapes;

        size_t coopmat_shape_space;
        if (options.optimizationLevel >= 5) {
          coopmat_shape_space = 3;
        } else {
          coopmat_shape_space = 1;
        }

        for (const denox::CoopmatShape &shape :
             options.deviceInfo.coopmat.shapes) {
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
        std::ranges::stable_sort(coopmatShapes,
                                 [](const auto &lhs, const auto &rhs) {
                                   return lhs.first > rhs.first;
                                 });
        coopmatShapes.resize(
            std::min<size_t>(coopmatShapes.size(), coopmat_shape_space));
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

            const uint32_t acc_register_estimate = (cm_m * cm_n) / subgroupSize;

            const uint32_t A_a_register_estimate =
                (cm_m * a_cm_k) / subgroupSize;
            const uint32_t A_b_register_estimate =
                (a_cm_k * cm_n) / subgroupSize;

            const uint32_t B_a_register_estimate =
                (cm_m * b_cm_k) / subgroupSize;
            const uint32_t B_b_register_estimate =
                (b_cm_k * cm_n) / subgroupSize;

            for (uint32_t wg_m = 1; wg_m < 16; ++wg_m) {
              for (uint32_t wg_n = 1; wg_n < 16; ++wg_n) {
                const uint32_t workgroup_size = wg_m * wg_n * subgroupSize;
                if (workgroup_size < 128 ||
                    workgroup_size > options.deviceInfo.limits
                                         .maxComputeWorkGroupInvocations) {
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
                        if (A_prefetch_A_QQ % wg_n != 0) {
                          continue;
                        }

                        const uint32_t B_prefetch_A_QQ =
                            (cm_m * b_cm_k * b_sg_k * sg_m) / 8;
                        if (B_prefetch_A_QQ % wg_n != 0) {
                          continue;
                        }

                        const uint32_t A_prefetch_A_SQQ =
                            A_prefetch_A_QQ / wg_n;

                        const uint32_t B_prefetch_A_SQQ =
                            B_prefetch_A_QQ / wg_n;
                        // 16bytes word fetched per invocation!
                        const uint32_t A_prefetch_A_IQQ =
                            (A_prefetch_A_SQQ + subgroupSize - 1) /
                            subgroupSize;
                        const uint32_t B_prefetch_A_IQQ =
                            (B_prefetch_A_SQQ + subgroupSize - 1) /
                            subgroupSize;

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

                        const uint32_t A_prefetch_B_SQQ =
                            A_prefetch_B_QQ / wg_m;
                        const uint32_t B_prefetch_B_SQQ =
                            B_prefetch_B_QQ / wg_m;

                        const uint32_t A_prefetch_B_IQQ =
                            (A_prefetch_B_SQQ + subgroupSize - 1) /
                            subgroupSize;

                        const uint32_t B_prefetch_B_IQQ =
                            (B_prefetch_B_SQQ + subgroupSize - 1) /
                            subgroupSize;

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

                        // register estimate is only proportional to the
                        // register counts, so there is a good chance that 160
                        // estimate corresponds to only 50-60 live registers at
                        // a time.
                        if (register_estimate > 200) {
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
                            static_cast<double>(options.deviceInfo.limits
                                                    .maxComputeSharedMemory) *
                                WG_SH_OCCUPANCY) {
                          continue;
                        }

                        if (options.optimizationLevel > 2) {
                          m_configs.push_back(ConvConvAddConfig{
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
                              .subgroupSize = subgroupSize,
                          });
                        }

                        if (options.optimizationLevel > 3) {

                          m_configs.push_back(ConvConvAddConfig{
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
                              .b_async = false,
                              .subgroupSize = subgroupSize,
                          });
                          m_configs.push_back(ConvConvAddConfig{
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
                              .b_async = true,
                              .subgroupSize = subgroupSize,
                          });
                        }

                        m_configs.push_back(ConvConvAddConfig{
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
                            .subgroupSize = subgroupSize,
                        });
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (m_configs.empty()) {
        std::cerr << "Warning: ConcatConvCMShader: Failed to find any valid "
                     "configuration."
                  << std::endl;
      }
    }
  } else {
    // === Read valid configurations from file ====
    auto fd = io::File::open(
        io::Path::assets() / "compiler/src/denox/compiler/implement/shaders/"
                             "conv/conv_conv_add_cm.configs",
        io::File::OpenMode::Read);
    std::string str;
    str.resize(fd.size());
    fd.read_exact(std::span<std::byte>(
        reinterpret_cast<std::byte *>(str.data()), str.size()));
    std::stringstream ss(str);

    ConvConvAddConfig config;
    while (ss >> config.cm_m >> config.a_cm_k >> config.b_cm_k >> config.cm_n >>
           config.sg_m >> config.a_sg_k >> config.b_sg_k >> config.sg_n >>
           config.wg_m >> config.wg_n >> config.a_async >> config.b_async >>
           config.subgroupSize) {

      // trivial workgroup size checks (should basically never fail)
      if (config.subgroupSize >
          options.deviceInfo.limits.maxComputeWorkGroupSize[0]) {
        continue;
      }
      const uint32_t sg_count = config.wg_m * config.wg_n;
      if (sg_count > options.deviceInfo.limits.maxComputeWorkGroupSize[1]) {
        continue;
      }
      if (1 > options.deviceInfo.limits.maxComputeWorkGroupSize[2]) {
        continue;
      }
      const uint32_t wg_size = sg_count * config.subgroupSize;
      if (wg_size > options.deviceInfo.limits.maxComputeWorkGroupInvocations) {
        continue;
      }

      // check if valid on this device!
      if (options.deviceInfo.subgroup.controlProperties.supported) {
        const bool sg_supported =
            std::ranges::count(options.deviceInfo.subgroup.controlProperties
                                   .supportedSubgroupSizes,
                               config.subgroupSize) != 0;
        if (!sg_supported) {
          continue; // subgroup control available, but subgroupSize invalid.
        }
      } else {
        if (config.subgroupSize != options.deviceInfo.subgroup.subgroupSize) {
          continue; // invalid subgroup size.
        }
      }

      if ((config.cm_m % 8 != 0) || (config.a_cm_k % 8 != 0) ||
          (config.b_cm_k % 8 != 0) || (config.cm_n % 8 != 0)) {
        continue;
      }

      bool a_shape_supported = false;
      for (const denox::CoopmatShape &shape :
           options.deviceInfo.coopmat.shapes) {
        if (shape.atype == memory::Dtype::F16 &&
            shape.btype == memory::Dtype::F16 &&
            shape.ctype == memory::Dtype::F16 &&
            shape.acctype == memory::Dtype::F16 && shape.subgroupScope &&
            shape.M == config.cm_m && shape.K == config.a_cm_k &&
            shape.N == config.cm_n) {
          a_shape_supported = true;
          break;
        }
      }
      bool b_shape_supported = false;
      for (const denox::CoopmatShape &shape :
           options.deviceInfo.coopmat.shapes) {
        if (shape.atype == memory::Dtype::F16 &&
            shape.btype == memory::Dtype::F16 &&
            shape.ctype == memory::Dtype::F16 &&
            shape.acctype == memory::Dtype::F16 && shape.subgroupScope &&
            shape.M == config.cm_m && shape.K == config.b_cm_k &&
            shape.N == config.cm_n) {
          b_shape_supported = true;
          break;
        }
      }
      if (!a_shape_supported || !b_shape_supported) {
        continue;
      }

      const uint32_t acc_register_estimate =
          (config.cm_m * config.cm_n) / config.subgroupSize;

      const uint32_t A_a_register_estimate =
          (config.cm_m * config.a_cm_k) / config.subgroupSize;
      const uint32_t A_b_register_estimate =
          (config.a_cm_k * config.cm_n) / config.subgroupSize;

      const uint32_t B_a_register_estimate =
          (config.cm_m * config.b_cm_k) / config.subgroupSize;
      const uint32_t B_b_register_estimate =
          (config.b_cm_k * config.cm_n) / config.subgroupSize;

      const uint32_t coopmats_register_estimate =
          acc_register_estimate * config.sg_n * config.sg_m +
          std::min(A_a_register_estimate * config.sg_m +
                       A_b_register_estimate * config.sg_n,
                   B_a_register_estimate * config.sg_m * B_b_register_estimate *
                       config.sg_n);

      const uint32_t A_prefetch_A_QQ =
          (config.cm_m * config.a_cm_k * config.a_sg_k * config.sg_m) / 8;
      if (A_prefetch_A_QQ % config.wg_n != 0) {
        continue;
      }

      const uint32_t B_prefetch_A_QQ =
          (config.cm_m * config.b_cm_k * config.b_sg_k * config.sg_m) / 8;
      if (B_prefetch_A_QQ % config.wg_n != 0) {
        continue;
      }

      const uint32_t A_prefetch_A_SQQ = A_prefetch_A_QQ / config.wg_n;

      const uint32_t B_prefetch_A_SQQ = B_prefetch_A_QQ / config.wg_n;
      // 16bytes word fetched per invocation!
      const uint32_t A_prefetch_A_IQQ =
          (A_prefetch_A_SQQ + config.subgroupSize - 1) / config.subgroupSize;
      const uint32_t B_prefetch_A_IQQ =
          (B_prefetch_A_SQQ + config.subgroupSize - 1) / config.subgroupSize;

      const uint32_t A_prefetch_B_QQ =
          (config.a_cm_k * config.cm_n * config.a_sg_k * config.sg_n) / 8;
      if (A_prefetch_B_QQ % config.wg_m != 0) {
        continue; // uneven load balancing between subgroups.
      }

      const uint32_t B_prefetch_B_QQ =
          (config.b_cm_k * config.cm_n * config.b_sg_k * config.sg_n) / 8;
      if (B_prefetch_B_QQ % config.wg_m != 0) {
        continue; // uneven load balancing between subgroups.
      }

      const uint32_t A_prefetch_B_SQQ = A_prefetch_B_QQ / config.wg_m;
      const uint32_t B_prefetch_B_SQQ = B_prefetch_B_QQ / config.wg_m;

      const uint32_t A_prefetch_B_IQQ =
          (A_prefetch_B_SQQ + config.subgroupSize - 1) / config.subgroupSize;

      const uint32_t B_prefetch_B_IQQ =
          (B_prefetch_B_SQQ + config.subgroupSize - 1) / config.subgroupSize;

      const uint32_t A_prefetch_A_register_estimate =
          A_prefetch_A_IQQ * 4; // uvec4
      const uint32_t A_prefetch_B_register_estimate =
          A_prefetch_B_IQQ * 4; // uvec4

      const uint32_t B_prefetch_A_register_estimate =
          B_prefetch_A_IQQ * 4; // uvec4
      const uint32_t B_prefetch_B_register_estimate =
          B_prefetch_B_IQQ * 4; // uvec4

      const uint32_t A_prefetch_register_estimate =
          A_prefetch_A_register_estimate + A_prefetch_B_register_estimate;

      const uint32_t B_prefetch_register_estimate =
          B_prefetch_A_register_estimate + B_prefetch_B_register_estimate;

      const uint32_t register_estimate =
          coopmats_register_estimate +
          std::max(A_prefetch_register_estimate, B_prefetch_register_estimate);

      // register estimate is only proportional to the
      // register counts, so there is a good chance that 160
      // estimate corresponds to only 50-60 live registers at
      // a time.
      if (register_estimate > 200) {
        continue; // to many registers (conservative limit,
                  // because optimizers might reduce this
                  // drastically)
      }

      const uint32_t A_sh_a_size = (config.wg_m * config.cm_m * config.a_cm_k *
                                    config.a_sg_k * config.sg_m) *
                                   2;

      const uint32_t B_sh_a_size = (config.wg_m * config.cm_m * config.b_cm_k *
                                    config.b_sg_k * config.sg_m) *
                                   2;

      const uint32_t A_sh_b_size = (config.wg_n * config.a_cm_k * config.cm_n *
                                    config.a_sg_k * config.sg_n) *
                                   2;

      const uint32_t B_sh_b_size = (config.wg_n * config.b_cm_k * config.cm_n *
                                    config.b_sg_k * config.sg_n) *
                                   2;

      const uint32_t sh_out_size = config.wg_m * config.wg_n * config.sg_m *
                                   config.sg_n * config.cm_m * config.cm_n * 2;

      const uint32_t A_sh_size =
          std::max(A_sh_a_size + A_sh_b_size, sh_out_size);

      const uint32_t B_sh_size =
          std::max(B_sh_a_size + B_sh_b_size, sh_out_size);

      const uint32_t sh_size = std::max(A_sh_size, B_sh_size);

      if (sh_size > options.deviceInfo.limits.maxComputeSharedMemory) {
        continue;
      }

      m_configs.push_back(config);
    }
  }
  assert(!m_configs.empty());

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

  const auto isAdd = [](const ComputeOp &op) -> bool {
    return op.tag() == ComputeOpKind::Add;
  };
  const auto convSupported = [](const ComputeOp &op) -> bool {
    if (op.tag() != ComputeOpKind::Conv) {
      return false;
    }
    // NOTE: not yet properly constrained !!!
    return true;
  };

  {
    Pattern conv_conv_add_pattern;
    Pattern::EP add = conv_conv_add_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP out = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP a = convA->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP b = convB->matchSrc(0);

    add->matchRank(2);
    add->matchValue(isAdd);
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_conv_conv_add_patternn =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out);
    m_capabilities.patterns.emplace_back(std::move(conv_conv_add_pattern),
                                         std::move(a), std::move(b),
                                         std::move(out));
  }

  if (options.features.enableConvReluFusion) {
    Pattern conv_conv_add_activation_pattern;
    Pattern::EP add = conv_conv_add_activation_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP sum = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP a = convA->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP b = convB->matchSrc(0);
    Pattern::EP activation = sum->matchOutgoing();
    Pattern::NP out = activation->matchDst();

    add->matchRank(2);
    add->matchValue(isAdd);
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);
    activation->matchRank(1);
    activation->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return func.kind() == ActivationFunctionKind::ReLU ||
             func.kind() == ActivationFunctionKind::LeakyReLU;
    });

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_conv_conv_add_activation_pattern =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out,
                                  memory::nullopt, activation);
    m_capabilities.patterns.emplace_back(
        std::move(conv_conv_add_activation_pattern), std::move(a),
        std::move(b), std::move(out));
  }

  if (options.features.enableUpsampleConvFusion) {
    Pattern A_upsample_conv_conv_add_pattern;
    Pattern::EP add = A_upsample_conv_conv_add_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP out = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP upsampledA = convA->matchSrc(0);
    Pattern::EP upsample = upsampledA->matchIncoming();
    Pattern::NP a = upsample->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP b = convB->matchSrc(0);

    add->matchRank(2);
    add->matchValue(isAdd);
    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      return op.upsample().scalingFactor == 2 &&
             op.upsample().mode == FilterMode::Nearest;
    });
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_A_upsample_conv_conv_add_pattern =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out,
                                  upsample, memory::nullopt);
    m_capabilities.patterns.emplace_back(
        std::move(A_upsample_conv_conv_add_pattern), std::move(a),
        std::move(b), std::move(out));
  }

  if (options.features.enableUpsampleConvFusion) {
    Pattern B_upsample_conv_conv_add_pattern;
    Pattern::EP add = B_upsample_conv_conv_add_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP out = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP a = convA->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP upsampledB = convB->matchSrc(0);
    Pattern::EP upsample = upsampledB->matchIncoming();
    Pattern::NP b = upsample->matchSrc(0);

    add->matchRank(2);
    add->matchValue(isAdd);
    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      return op.upsample().scalingFactor == 2 &&
             op.upsample().mode == FilterMode::Nearest;
    });
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_B_upsample_conv_conv_add_pattern =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out,
                                  upsample, memory::nullopt);
    m_capabilities.patterns.emplace_back(
        std::move(B_upsample_conv_conv_add_pattern), std::move(a),
        std::move(b), std::move(out));
  }

  if (options.features.enableUpsampleConvFusion &&
      options.features.enableConvReluFusion) {
    Pattern A_upsample_conv_conv_add_activation_pattern;
    Pattern::EP add =
        A_upsample_conv_conv_add_activation_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP sum = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP upsampledA = convA->matchSrc(0);
    Pattern::EP upsample = upsampledA->matchIncoming();
    Pattern::NP a = upsample->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP b = convB->matchSrc(0);
    Pattern::EP activation = sum->matchOutgoing();
    Pattern::NP out = activation->matchDst();

    add->matchRank(2);
    add->matchValue(isAdd);
    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      return op.upsample().scalingFactor == 2 &&
             op.upsample().mode == FilterMode::Nearest;
    });
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);
    activation->matchRank(1);
    activation->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return func.kind() == ActivationFunctionKind::ReLU ||
             func.kind() == ActivationFunctionKind::LeakyReLU;
    });

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_A_upsample_conv_conv_add_activation_pattern =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out,
                                  upsample, activation);
    m_capabilities.patterns.emplace_back(
        std::move(A_upsample_conv_conv_add_activation_pattern), std::move(a),
        std::move(b), std::move(out));
  }

  if (options.features.enableUpsampleConvFusion &&
      options.features.enableConvReluFusion) {
    Pattern B_upsample_conv_conv_add_activation_pattern;
    Pattern::EP add =
        B_upsample_conv_conv_add_activation_pattern.matchEdge();
    Pattern::NP lhs = add->matchSrc(0);
    Pattern::NP rhs = add->matchSrc(1);
    Pattern::NP sum = add->matchDst();
    Pattern::EP convA = lhs->matchIncoming();
    Pattern::NP a = convA->matchSrc(0);
    Pattern::EP convB = rhs->matchIncoming();
    Pattern::NP upsampledB = convB->matchSrc(0);
    Pattern::EP upsample = upsampledB->matchIncoming();
    Pattern::NP b = upsample->matchSrc(0);
    Pattern::EP activation = sum->matchOutgoing();
    Pattern::NP out = activation->matchDst();

    add->matchRank(2);
    add->matchValue(isAdd);
    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      return op.upsample().scalingFactor == 2 &&
             op.upsample().mode == FilterMode::Nearest;
    });
    convA->matchRank(1);
    convA->matchValue(convSupported);
    convB->matchRank(1);
    convB->matchValue(convSupported);
    activation->matchRank(1);
    activation->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return func.kind() == ActivationFunctionKind::ReLU ||
             func.kind() == ActivationFunctionKind::LeakyReLU;
    });

    a->matchValue(tensorSupported);
    b->matchValue(tensorSupported);
    out->matchValue(tensorSupported);

    m_B_upsample_conv_conv_add_activation_pattern =
        static_cast<unsigned int>(m_patternHandles.size());
    m_patternHandles.emplace_back(a, convA, lhs, b, convB, rhs, add, out,
                                  upsample, activation);
    m_capabilities.patterns.emplace_back(
        std::move(B_upsample_conv_conv_add_activation_pattern), std::move(a),
        std::move(b), std::move(out));
  }
 }

std::size_t ConvConvAddCMShader::parameterMemorySize(
    const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &convA_pattern = m_patternHandles[pattern].convA;
  const auto &convB_pattern = m_patternHandles[pattern].convB;
  size_t elemCount = 0;

  memory::EdgeId convA_id = match[convA_pattern];
  const ComputeOp &convA_op = graph.get(convA_id);
  assert(convA_op.tag() == ComputeOpKind::Conv);
  const auto &convA = convA_op.conv();
  elemCount += convA->W->shape().elemCount();
  if (convA->B != nullptr) {
    elemCount += convA->B->shape();
  }

  memory::EdgeId convB_id = match[convB_pattern];
  const ComputeOp &convB_op = graph.get(convB_id);
  assert(convB_op.tag() == ComputeOpKind::Conv);
  const auto &convB = convB_op.conv();
  elemCount += convB->W->shape().elemCount();
  if (convB->B != nullptr) {
    elemCount += convB->B->shape();
  }

  return elemCount * memory::Dtype::F16.size();
}

memory::vector<unsigned int> ConvConvAddCMShader::acceptMatch(
    [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
        &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  [[maybe_unused]] const auto &a = opGraph.get(match[patternHandles.a]);
  [[maybe_unused]] const auto &b = opGraph.get(match[patternHandles.b]);

  [[maybe_unused]] const auto &convA =
      opGraph.get(match[patternHandles.convA]).conv();
  [[maybe_unused]] const auto &convB =
      opGraph.get(match[patternHandles.convB]).conv();

  [[maybe_unused]] const auto &out = opGraph.get(match[patternHandles.out]);

  const uint32_t A_C = static_cast<uint32_t>(a.channels.constant());
  const uint32_t B_C = static_cast<uint32_t>(b.channels.constant());
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());
  const uint32_t A_R = convA->W->shape().r;
  const uint32_t A_S = convA->W->shape().s;
  const uint32_t B_R = convB->W->shape().r;
  const uint32_t B_S = convB->W->shape().s;

  memory::vector<unsigned int> promissing;
  for (uint32_t c = 0; c < m_configs.size(); ++c) {

    // static constexpr size_t KK_ASYNC_LIMIT = 3;
    static constexpr size_t MAX_CHANNEL_TILE_OVERALLOCATION = 2;
    static constexpr size_t MAX_KTILE_OVERALLOCATION = 2;
    const auto &config = m_configs[c];

    const uint32_t A_RSC = A_R * A_S * A_C;
    const uint32_t B_RSC = B_R * B_S * B_C;

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
    const uint32_t wgSize = config.wg_m * config.wg_n * config.subgroupSize;
    if (wgSize < 128 || wgSize > 512) {
      continue;
    }

    if (m_optimizationLevel <= 2) {
      if (config.a_async != config.b_async) {
        continue;
      }
      if (config.a_cm_k != config.b_cm_k) {
        continue;
      }
      if (config.a_sg_k != config.b_sg_k) {
        continue;
      }
    }
    promissing.push_back(c);
  }
  return promissing;
}

static spirv::GlslCompilerInstance direct_conv_cm_compile(
    spirv::GlslCompiler *compiler, const io::Path &srcPath,
    unsigned int subgroupSize, unsigned int A_C, unsigned int B_C,
    unsigned int K, TensorFormat A_inputFormat, TensorFormat B_inputFormat,
    TensorFormat outputFormat,
    memory::optional<ActivationFunction> activationFunction,
    memory::uvec2 A_kernelSize, memory::uvec2 A_padding, memory::uvec2 A_stride,
    memory::uvec2 B_kernelSize, memory::uvec2 B_padding, memory::uvec2 B_stride,
    bool bias, const ConvConvAddConfig &config, uint32_t A_scalingFactor,
    uint32_t B_scalingFactor, bool subgroupControl,
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

  shader.define("A_SCALING_FACTOR", A_scalingFactor);
  shader.define("B_SCALING_FACTOR", B_scalingFactor);

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

  shader.define("A_KERNEL_X", A_kernelSize.x);
  shader.define("A_KERNEL_Y", A_kernelSize.y);
  shader.define("A_STRIDE_X", A_stride.x);
  shader.define("A_STRIDE_Y", A_stride.y);
  shader.define("A_PADDING_X", A_padding.x);
  shader.define("A_PADDING_Y", A_padding.y);

  shader.define("B_KERNEL_X", B_kernelSize.x);
  shader.define("B_KERNEL_Y", B_kernelSize.y);
  shader.define("B_STRIDE_X", B_stride.x);
  shader.define("B_STRIDE_Y", B_stride.y);
  shader.define("B_PADDING_X", B_padding.x);
  shader.define("B_PADDING_Y", B_padding.y);

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

  if (subgroupControl) {
    shader.define("SG_CONTROL");
  } else {
    shader.define("NSG_CONTROL");
  }

  return shader;
}

void ConvConvAddCMShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    [[maybe_unused]] unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {

  const ConvConvAddConfig &config = m_configs[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::EdgeId convA_id = match[patternHandles.convA];
  memory::EdgeId convB_id = match[patternHandles.convB];
  memory::NodeId aId = match[patternHandles.a];
  memory::NodeId bId = match[patternHandles.b];

  memory::NodeId outId = match[patternHandles.out];

  const ComputeOp &convA_op = opGraph.get(convA_id);
  const ComputeOp &convB_op = opGraph.get(convB_id);

  const auto &a = opGraph.get(aId);
  const auto &b = opGraph.get(bId);
  const auto &out = opGraph.get(outId);
  assert(convA_op.tag() == ComputeOpKind::Conv);
  assert(convB_op.tag() == ComputeOpKind::Conv);
  assert(a.channels.isConstant());
  assert(b.channels.isConstant());
  assert(out.channels.isConstant());
  assert(a.type == b.type);

  const ComputeOpConv &convA = convA_op.conv();
  const ComputeOpConv &convB = convB_op.conv();

  uint32_t A_scalingFactor = 1;
  if (pattern == m_A_upsample_conv_conv_add_activation_pattern ||
      pattern == m_A_upsample_conv_conv_add_pattern) {
    assert(patternHandles.upsample.has_value());
    A_scalingFactor =
        opGraph.get(match[*patternHandles.upsample]).upsample().scalingFactor;
  }

  uint32_t B_scalingFactor = 1;
  if (pattern == m_B_upsample_conv_conv_add_activation_pattern ||
      pattern == m_B_upsample_conv_conv_add_pattern) {
    assert(patternHandles.upsample.has_value());
    B_scalingFactor =
        opGraph.get(match[*patternHandles.upsample]).upsample().scalingFactor;
  }
  assert(B_scalingFactor == 1);

  memory::optional<ActivationFunction> activationFunction;
  if (patternHandles.relu.has_value()) {
    activationFunction =
        opGraph.get(match[*patternHandles.relu]).activation().func;
  }

  const uint32_t A_C = static_cast<uint32_t>(a.channels.constant());
  const uint32_t B_C = static_cast<uint32_t>(b.channels.constant());
  const uint32_t K = static_cast<uint32_t>(out.channels.constant());

  const uint32_t A_R = convA->W->shape().r;
  const uint32_t A_S = convA->W->shape().s;
  memory::uvec2 A_kernelSize{A_R, A_S};

  const uint32_t B_R = convB->W->shape().r;
  const uint32_t B_S = convB->W->shape().s;
  memory::uvec2 B_kernelSize{B_R, B_S};

  // dummy values; overwritten in direct_conv_cm_compile
  memory::FilterLayout A_filterLayout = memory::FilterLayout::RSCK;
  memory::FilterLayout B_filterLayout = memory::FilterLayout::RSCK;
  memory::BiasLayout biasLayout = memory::BiasLayout::C;

  bool useBias = (convA->B != nullptr) || (convB->B != nullptr);

  auto shader = direct_conv_cm_compile(
      m_compiler, m_srcPath, config.subgroupSize, A_C, B_C, K, //
      a.format, b.format, out.format,                          //
      activationFunction,                                      //
      A_kernelSize, convA->padding, convA->stride,             //
      B_kernelSize, convB->padding, convB->stride,             //
      useBias, config,                                         //
      A_scalingFactor, B_scalingFactor, m_subgroupControl,     //
      &A_filterLayout, &B_filterLayout, &biasLayout);

  std::uint32_t tileX = config.cm_n * config.sg_n * config.wg_n;
  std::uint32_t tileY = config.cm_m;
  std::uint32_t tileZ = config.sg_m * config.wg_m;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX, false, false);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileY, false, false);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileZ, false, false);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  if (m_subgroupControl) {
    dispatch.setFixedSubgroupSize(config.subgroupSize);
  }

  TensorId A_W = impl.createParameter( //
      A_filterLayout.size(convA->W->shape()) * memory::Dtype::F16.size(),
      TensorDataType::Float16, TensorStorage::StorageBuffer,
      TensorFormat::Optimal,
      [W = convA->W,
       filterLayout = A_filterLayout]() -> std::vector<std::byte> {
        memory::FilterTensor filter{
            {W->shape(), filterLayout, memory::Dtype::F16}, W->const_view()};
        std::vector<std::byte> raw{filter.span().begin(), filter.span().end()};
        return raw;
      });

  TensorId B_W = impl.createParameter( //
      B_filterLayout.size(convB->W->shape()) * memory::Dtype::F16.size(),
      TensorDataType::Float16, TensorStorage::StorageBuffer,
      TensorFormat::Optimal,
      [W = convB->W,
       filterLayout = B_filterLayout]() -> std::vector<std::byte> {
        memory::FilterTensor filter{
            {W->shape(), filterLayout, memory::Dtype::F16}, W->const_view()};
        std::vector<std::byte> raw{filter.span().begin(), filter.span().end()};
        return raw;
      });

  memory::optional<TensorId> bias = memory::nullopt;
  if (convA->B != nullptr && convB->B != nullptr) {
    assert(convA->B->shape() == convB->B->shape());
    bias = impl.createParameter(
        biasLayout.size(convA->B->shape()) * memory::Dtype::F16.size(),
        TensorDataType::Float16, TensorStorage::StorageBuffer,
        TensorFormat::Optimal,
        [A_bias = convA->B, B_bias = convB->B,
         biasLayout]() -> std::vector<std::byte> {
          unsigned int K = A_bias->shape();
          memory::BiasTensor tensor{{
              K,
              biasLayout,
              memory::Dtype::F16,
          }};
          for (unsigned int k = 0; k < K; ++k) {
            tensor.at(k) = static_cast<memory::f64>(A_bias->at(k)) +
                           static_cast<memory::f64>(B_bias->at(k));
          }
          std::vector<std::byte> raw(tensor.span().begin(),
                                     tensor.span().end());
          return raw;
        });
  } else if (convA->B != nullptr) {
    bias = impl.createParameter(
        biasLayout.size(convA->B->shape()) * memory::Dtype::F16.size(),
        TensorDataType::Float16, TensorStorage::StorageBuffer,
        TensorFormat::Optimal,
        [B = convA->B, biasLayout]() -> std::vector<std::byte> {
          memory::BiasTensor tensor{
              {B->shape(), biasLayout, memory::Dtype::F16}, B->const_view()};
          std::vector<std::byte> raw(tensor.span().begin(),
                                     tensor.span().end());
          return raw;
        });
  } else if (convB->B != nullptr) {
    bias = impl.createParameter(
        biasLayout.size(convB->B->shape()) * memory::Dtype::F16.size(),
        TensorDataType::Float16, TensorStorage::StorageBuffer,
        TensorFormat::Optimal,
        [B = convB->B, biasLayout]() -> std::vector<std::byte> {
          memory::BiasTensor tensor{
              {B->shape(), biasLayout, memory::Dtype::F16}, B->const_view()};
          std::vector<std::byte> raw(tensor.span().begin(),
                                     tensor.span().end());
          return raw;
        });
  }

  dispatch.addBinding("A_SET", "A_BINDING", Access::ReadOnly, aId);
  dispatch.addBinding("B_SET", "B_BINDING", Access::ReadOnly, bId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);

  dispatch.addParamBinding("A_FILTER_SET", "A_FILTER_BINDING", A_W);
  dispatch.addParamBinding("B_FILTER_SET", "B_FILTER_BINDING", B_W);
  if (bias) {
    dispatch.addParamBinding("BIAS_SET", "BIAS_BINDING", *bias);
  }

  dispatch.addPushConstant(
      PushConstant::Dynamic(out.width, memory::Dtype::U32));
  dispatch.addPushConstant(
      PushConstant::Dynamic(out.height, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(a.width, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(a.height, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(b.width, memory::Dtype::U32));
  dispatch.addPushConstant(PushConstant::Dynamic(b.height, memory::Dtype::U32));

  dispatch.setSourcePath(m_srcPath);

  Sym areads = symGraph.mul(symGraph.mul(a.width, a.height),
                            A_C * size_of(TensorDataType::Float16));
  Sym breads = symGraph.mul(symGraph.mul(b.width, b.height),
                            B_C * size_of(TensorDataType::Float16));

  Sym inreads = symGraph.add(areads, breads);
  size_t wreads = 0;
  wreads += convA->W->byteSize();
  wreads += convB->W->byteSize();
  if (convA->B) {
    wreads += convA->B->byteSize();
  }
  if (convB->B) {
    wreads += convB->B->byteSize();
  }

  Sym reads = symGraph.add(wreads, inreads);

  Sym writes = symGraph.mul(symGraph.mul(out.width, out.height),
                            K * size_of(TensorDataType::Float16));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);

  dispatch.useCoopmatShape(CoopmatShape{
      .M = config.cm_m,
      .N = config.cm_n,
      .K = config.a_cm_k,
      .atype = memory::Dtype::F16,
      .btype = memory::Dtype::F16,
      .ctype = memory::Dtype::F16,
      .acctype = memory::Dtype::F16,
      .saturatingAccumulation = true,
      .subgroupScope = true,
  });
  if (config.a_cm_k != config.b_cm_k) {
    dispatch.useCoopmatShape(CoopmatShape{
        .M = config.cm_m,
        .N = config.cm_n,
        .K = config.b_cm_k,
        .atype = memory::Dtype::F16,
        .btype = memory::Dtype::F16,
        .ctype = memory::Dtype::F16,
        .acctype = memory::Dtype::F16,
        .saturatingAccumulation = true,
        .subgroupScope = true,
    });
  }
  {
    const uint32_t A_sh_a_size = (config.wg_m * config.cm_m * config.a_cm_k *
                                  config.a_sg_k * config.sg_m) *
                                 2;
    const uint32_t B_sh_a_size = (config.wg_m * config.cm_m * config.b_cm_k *
                                  config.b_sg_k * config.sg_m) *
                                 2;
    const uint32_t A_sh_b_size = (config.wg_n * config.a_cm_k * config.cm_n *
                                  config.a_sg_k * config.sg_n) *
                                 2;
    const uint32_t B_sh_b_size = (config.wg_n * config.b_cm_k * config.cm_n *
                                  config.b_sg_k * config.sg_n) *
                                 2;
    const uint32_t sh_out_size = config.wg_m * config.wg_n * config.sg_m *
                                 config.sg_n * config.cm_m * config.cm_n * 2;
    const uint32_t A_sh_size = std::max(A_sh_a_size + A_sh_b_size, sh_out_size);

    const uint32_t B_sh_size = std::max(B_sh_a_size + B_sh_b_size, sh_out_size);

    const uint32_t sh_size = std::max(A_sh_size, B_sh_size);
    dispatch.setSharedMemory(sh_size);
  }

  Sym OHW = symGraph.mul(out.width, out.height);
  Sym convA_flops = symGraph.mul(OHW, A_C * K * A_R * A_S);
  Sym convB_flops = symGraph.mul(OHW, B_C * K * B_R * B_S);

  dispatch.setFlops(symGraph.add(convA_flops, convB_flops));

  memory::string inputA = "x";
  if (A_scalingFactor != 1) {
    inputA = fmt::format(
        "upsample(x,mode=nearest,scaling_factor={})", A_scalingFactor);
  }
  memory::string inputB = "y";
  if (B_scalingFactor != 1) {
    inputB = fmt::format(
        "upsample(y,mode=nearest,scaling_factor={})", B_scalingFactor);
  }
  memory::string convAOperation = fmt::format(
      "conv2d({},kernel_size=({},{}),bias={},stride=({},{}),padding=({},{}),"
      "dialation=(1,1))",
      inputA, convA->W->shape().s, convA->W->shape().r, convA->B != nullptr,
      convA->stride.x, convA->stride.y, convA->padding.x, convA->padding.y);
  memory::string convBOperation = fmt::format(
      "conv2d({},kernel_size=({},{}),bias={},stride=({},{}),padding=({},{}),"
      "dialation=(1,1))",
      inputB, convB->W->shape().s, convB->W->shape().r, convB->B != nullptr,
      convB->stride.x, convB->stride.y, convB->padding.x, convB->padding.y);
  memory::string operation =
      fmt::format("{}+{}", convAOperation, convBOperation);
  if (activationFunction) {
    switch (activationFunction->kind()) {
    case ActivationFunctionKind::ReLU:
      operation = fmt::format("relu({})", operation);
      break;
    case ActivationFunctionKind::LeakyReLU:
      operation = fmt::format(
          "leaky_relu({},alpha={})", operation,
          activationFunction->leaky_relu().alpha);
      break;
    case ActivationFunctionKind::SiLU:
      operation = fmt::format("silu({})", operation);
      break;
    case ActivationFunctionKind::Swish:
      operation = fmt::format("swish({},beta={})", operation,
                              activationFunction->swish().beta);
      break;
    }
  }
  dispatch.setOperation(std::move(operation));

  dispatch.setConfig(
      fmt::format("CM_M={}#A_CM_K={}#B_CM_K={}#CM_N={}#SG_M={}#A_SG_K={}#B_SG_"
                  "K={}#SG_N={}#WG_M={}#WG_N={}#A_ASYNC={}#B_ASYNC={}",
                  config.cm_m, config.a_cm_k, config.b_cm_k, config.cm_n,
                  config.sg_m, config.a_sg_k, config.b_sg_k, config.sg_n,
                  config.wg_m, config.wg_n, config.a_async, config.b_async));

  dispatch.setName(name());
}
memory::string ConvConvAddCMShader::name() const { return "conv-conv-add-cm"; }
} // namespace denox::compiler::shaders
