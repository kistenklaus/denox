#include "denox/compiler/implement/shaders/activation/BasicActivationShader.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/dtype/dtype.hpp"

namespace denox::compiler::shaders {

BasicActivationShader::BasicActivationShader(spirv::GlslCompiler *compiler,
                                             const CompileOptions &options)
    : m_compiler(compiler),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize),
      m_optimizationLevel(options.optimizationLevel) {

  { // Generate HWC config space
    uint32_t min_invocC = 1;
    uint32_t max_invocC = 4;
    if (options.optimizationLevel < 4) {
      min_invocC = 2;
      max_invocC = 2;
    }

    uint32_t min_invocH = 1;
    uint32_t max_invocH = 8;
    if (options.optimizationLevel < 4) {
      min_invocH = 4;
      max_invocH = 4;
    }
    uint32_t min_wg_size = m_subgroupSize;
    uint32_t max_wg_size = std::max(
        static_cast<uint32_t>(algorithm::align_up(512, m_subgroupSize)),
        options.deviceInfo.limits.maxComputeWorkGroupInvocations);

    if (options.optimizationLevel < 4) {
      min_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
      max_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
    }

    for (uint32_t invocC = min_invocC; invocC <= max_invocC; ++invocC) {
      for (uint32_t invocH = min_invocH; invocH <= max_invocH; ++invocH) {
        for (uint32_t wgSize = min_wg_size; wgSize <= max_wg_size;
             wgSize += m_subgroupSize) {
          m_hwc_configs.push_back(Config_HWC{
              .invocC = invocC,
              .invocH = invocH,
              .wgSizeHint = wgSize,
          });
        }
      }
    }
  }
  { // HWC8 config space
    uint32_t min_invocH = 1;
    uint32_t max_invocH = 2;
    if (options.optimizationLevel < 4) {
      min_invocH = 1;
      max_invocH = 1;
    }

    uint32_t min_invocW = 1;
    uint32_t max_invocW = 2;
    if (options.optimizationLevel < 4) {
      min_invocW = 1;
      max_invocW = 1;
    }
    uint32_t min_wg_size = m_subgroupSize;
    uint32_t max_wg_size = std::max(
        static_cast<uint32_t>(algorithm::align_up(512, m_subgroupSize)),
        options.deviceInfo.limits.maxComputeWorkGroupInvocations);

    if (options.optimizationLevel < 4) {
      min_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
      max_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
    }

    for (uint32_t invocW = min_invocW; invocW <= max_invocW; ++invocW) {
      for (uint32_t invocH = min_invocH; invocH <= max_invocH; ++invocH) {

        for (uint32_t wgSize = min_wg_size; wgSize <= max_wg_size;
             wgSize += m_subgroupSize) {
          m_hwc8_configs.push_back(Config_HWC8{
              .invocW = invocW,
              .invocH = invocH,
              .wgSizeHint = wgSize,
          });
        }
      }
    }
  }
  { // CHWC8 config space
    uint32_t min_invocH = 1;
    uint32_t max_invocH = 2;
    if (options.optimizationLevel < 4) {
      min_invocH = 1;
      max_invocH = 1;
    }

    uint32_t min_invocW = 1;
    uint32_t max_invocW = 2;
    if (options.optimizationLevel < 4) {
      min_invocW = 1;
      max_invocW = 1;
    }

    uint32_t min_wg_size = m_subgroupSize;
    uint32_t max_wg_size = std::max(
        static_cast<uint32_t>(algorithm::align_up(512, m_subgroupSize)),
        options.deviceInfo.limits.maxComputeWorkGroupInvocations);

    if (options.optimizationLevel < 4) {
      min_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
      max_wg_size =
          static_cast<uint32_t>(algorithm::align_up(256u, m_subgroupSize));
    }
    for (uint32_t invocW = min_invocW; invocW <= max_invocW; ++invocW) {
      for (uint32_t invocH = min_invocH; invocH <= max_invocH; ++invocH) {
        for (uint32_t wgSize = min_wg_size; wgSize <= max_wg_size;
             wgSize += m_subgroupSize) {
          m_chwc8_configs.push_back(Config_CHWC8{
              .invocW = invocW,
              .invocH = invocH,
              .wgSize = wgSize,
          });
        }
      }
    }
  }

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    if (tensor.type != TensorDataType::Float16) {
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
    Pattern basicPattern;
    auto in = basicPattern.matchNode();
    auto acti = in->matchOutgoing();
    auto out = acti->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    acti->matchRank(1);
    acti->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return (func.kind() == ActivationFunctionKind::ReLU) ||
             (func.kind() == ActivationFunctionKind::LeakyReLU);
    });
    m_patternHandles.emplace_back(in, std::move(acti), out);
    m_capabilities.patterns.emplace_back(std::move(basicPattern), std::move(in),
                                         std::move(out));
  }
}

memory::vector<unsigned int> BasicActivationShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (in.format != out.format) {
    return {};
  }
  assert(in.channels.isConstant());

  if (in.type != TensorDataType::Float16) {
    diag::invalid_state();
  }
  if (out.type != TensorDataType::Float16) {
    diag::invalid_state();
  }

  if (in.channels != out.channels) {
    diag::invalid_state();
  }

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  enum Variant {
    HWC,
    HWC8,
    CHWC8,
  };
  bool vec = in.channels.constant() % 8 == 0;
  Variant variant = HWC;
  if (in.format == TensorFormat::SSBO_HWC && vec) {
    variant = HWC8;
  } else if (in.format == TensorFormat::SSBO_CHWC8) {
    assert(vec);
    variant = CHWC8;
  }

  memory::vector<unsigned int> promissing;
  switch (variant) {
  case HWC: {
    for (uint32_t c = 0; c < m_hwc_configs.size(); ++c) {
      const auto &config = m_hwc_configs[c];
      uint32_t wgC;
      if (C < 256) {
        wgC = (C + config.invocC - 1) / config.invocC;
      } else {
        wgC = 128;
      }
      uint32_t ctile = wgC * config.invocC;
      if (C >= 256 && ctile < C) {
        continue;
      }
      uint32_t wgH = 1;
      uint32_t wgW = (config.wgSizeHint + wgC * wgH - 1) / (wgC * wgH);
      uint32_t wgSize = wgC * wgW * wgH;
      if (wgSize < 64 ||
          wgSize > std::min(512u, m_maxComputeWorkGroupInvocations)) {
        continue;
      }
      if (wgSize % 32 != 0) {
        continue;
      }
      if (wgC >= m_maxComputeWorkGroupSize[0]) {
        continue;
      }
      if (wgW >= m_maxComputeWorkGroupSize[1]) {
        continue;
      }
      if (wgH >= m_maxComputeWorkGroupSize[2]) {
        continue;
      }

      if (m_optimizationLevel < 4) {
        if (std::popcount(wgSize) != 1) {
          continue; // only accept powers of 2
        }
      }

      promissing.push_back(c);
    }
    if (promissing.empty()) {
      for (uint32_t c = 0; c < m_hwc_configs.size(); ++c) {
        const auto &config = m_hwc_configs[c];
        uint32_t wgC;
        if (C < 256) {
          wgC = (C + config.invocC - 1) / config.invocC;
        } else {
          wgC = 128;
        }
        uint32_t ctile = wgC * config.invocC;
        if (C >= 256 && ctile < C) { // weaker condition
          continue;
        }
        uint32_t wgH = 1;
        uint32_t wgW = (config.wgSizeHint + wgC * wgH - 1) / (wgC * wgH);
        uint32_t wgSize = wgC * wgW * wgH;

        if (wgSize < m_subgroupSize ||
            wgSize >= m_maxComputeWorkGroupInvocations) {
          continue;
        }
        if (wgSize % 32 != 0) {
          continue;
        }

        if (wgC >= m_maxComputeWorkGroupSize[0]) {
          continue;
        }
        if (wgW >= m_maxComputeWorkGroupSize[1]) {
          continue;
        }
        if (wgH >= m_maxComputeWorkGroupSize[2]) {
          continue;
        }

        promissing.push_back(c);
      }
    }
    if (promissing.empty()) {
      for (uint32_t c = 0; c < m_hwc_configs.size(); ++c) {
        const auto &config = m_hwc_configs[c];
        uint32_t wgC;
        if (C < 256) {
          wgC = (C + config.invocC - 1) / config.invocC;
        } else {
          wgC = 128;
        }
        uint32_t ctile = wgC * config.invocC;
        if (C <= 256 && ctile < C) { // weaker condition
          continue;
        }
        uint32_t wgH = 1;
        uint32_t wgW = (config.wgSizeHint + wgC * wgH - 1) / (wgC * wgH);
        uint32_t wgSize = wgC * wgW * wgH;

        if (wgSize < m_subgroupSize ||
            wgSize >= m_maxComputeWorkGroupInvocations) {
          continue;
        }

        if (wgC >= m_maxComputeWorkGroupSize[0]) {
          continue;
        }
        if (wgW >= m_maxComputeWorkGroupSize[1]) {
          continue;
        }
        if (wgH >= m_maxComputeWorkGroupSize[2]) {
          continue;
        }

        promissing.push_back(c);
      }
    }
    assert(!promissing.empty());
    break;
  }
  case HWC8: {
    for (uint32_t c = 0; c < m_hwc8_configs.size(); ++c) {
      const auto &config = m_hwc8_configs[c];
      uint32_t wgC;
      if (C < std::min(m_subgroupSize * 8, 512u)) {
        assert(C % 8 == 0);
        wgC = C / 8;
      } else {
        wgC = 128u;
      }
      const uint32_t wgH = 1;
      uint32_t wgW = (config.wgSizeHint + wgC * wgH - 1) / (wgC * wgH);

      uint32_t wgSize = wgC * wgW * wgH;

      if (wgSize < m_subgroupSize ||
          wgSize >= m_maxComputeWorkGroupInvocations) {
        continue;
      }

      if (wgC >= m_maxComputeWorkGroupSize[0]) {
        continue;
      }
      if (wgW >= m_maxComputeWorkGroupSize[1]) {
        continue;
      }
      if (wgH >= m_maxComputeWorkGroupSize[2]) {
        continue;
      }
      if (m_optimizationLevel < 4 && std::popcount(wgSize) != 1) {
        continue;
      }
      if (m_optimizationLevel < 4 &&
          (wgSize < std::min(4 * m_subgroupSize, 128u) ||
           wgSize > std::max(512u, m_subgroupSize * 8))) {
        continue;
      }

      promissing.push_back(c);
    }

    if (promissing.empty()) {
      for (uint32_t c = 0; c < m_hwc8_configs.size(); ++c) {
        const auto &config = m_hwc8_configs[c];
        uint32_t wgC;
        if (C < std::min(m_subgroupSize * 8, 512u)) {
          assert(C % 8 == 0);
          wgC = C / 8;
        } else {
          wgC = 128u;
        }
        const uint32_t wgH = 1;
        uint32_t wgW = (config.wgSizeHint + wgC * wgH - 1) / (wgC * wgH);

        uint32_t wgSize = wgC * wgW * wgH;

        if (wgSize < m_subgroupSize ||
            wgSize > m_maxComputeWorkGroupInvocations) {
          continue;
        }

        if (wgC >= m_maxComputeWorkGroupSize[0]) {
          continue;
        }
        if (wgW >= m_maxComputeWorkGroupSize[1]) {
          continue;
        }
        if (wgH >= m_maxComputeWorkGroupSize[2]) {
          continue;
        }
        promissing.push_back(c);
      }
    }

    assert(!promissing.empty());
    break;
  }
  case CHWC8: {
    for (uint32_t c = 0; c < m_hwc8_configs.size(); ++c) {
      const auto &config = m_chwc8_configs[c];
      uint32_t wgW = config.wgSize;
      if (m_optimizationLevel < 4 && std::popcount(wgW) != 1) {
        continue;
      }
      promissing.push_back(c);
    }
    if (promissing.empty()) {
      for (uint32_t c = 0; c < m_hwc8_configs.size(); ++c) {
        promissing.push_back(c);
      }
    }
    assert(!promissing.empty());
    break;
  default:
    denox::diag::unreachable();
  }
  }
  return promissing;
}

static spirv::GlslCompilerInstance
basic_activation_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                         unsigned int subgroupSize, TensorFormat inputFormat,
                         TensorFormat outputFormat, unsigned int channels,
                         memory::Dtype atype,
                         ActivationFunction activationFunction,
                         const BasicActivationShader::Config &config) {
  if (atype != memory::Dtype::F16) {
    diag::invalid_state();
  }
  auto shader = compiler->read(srcPath);
  shader.define("SG_SIZE", subgroupSize);
  shader.define("CH", channels);
  shader.define("in_atype", "float16_t");
  shader.define("IN_ATYPE_SIZE", 2);
  shader.define("out_atype", "float16_t");
  shader.define("OUT_ATYPE_SIZE", 2);

  switch (activationFunction.kind()) {
  case ActivationFunctionKind::ReLU:
    shader.define("ACTIVATION_ReLU");
    break;
  case ActivationFunctionKind::LeakyReLU:
    shader.define("ACTIVATION_LeakyReLU");
    shader.define("ACTIVATION_LeakyReLU_alpha",
                  fmt::format("({}f)", activationFunction.leaky_relu().alpha));
    break;
  case ActivationFunctionKind::SiLU:
  case ActivationFunctionKind::Swish:
    diag::invalid_state();
  }

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC && (channels % 8 == 0) &&
      (config.invocC % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");
  } else if (inputFormat == TensorFormat::SSBO_HWC &&
             outputFormat == TensorFormat::SSBO_HWC) {
    if (channels % 8 == 0) {
      DENOX_WARN(
          "BasicActivationShader implements non vectorized layouts for format, "
          "which may be vectorized, this works, but is suboptimal!");
    }
    // HWC layout (slow path)
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);

    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
  } else if (inputFormat == TensorFormat::SSBO_CHWC8 &&
             outputFormat == TensorFormat::SSBO_CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    diag::invalid_state();
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  return shader;
}

void BasicActivationShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &acti = opGraph.get(match[patternHandles.acti]).activation();

  assert(in.channels.isConstant());
  assert(in.channels == out.channels);
  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  enum Variant {
    HWC,
    HWC8,
    CHWC8,
  };
  bool vec = in.channels.constant() % 8 == 0;
  Variant variant = HWC;
  if (in.format == TensorFormat::SSBO_HWC && vec) {
    variant = HWC8;
  } else if (in.format == TensorFormat::SSBO_CHWC8) {
    assert(vec);
    variant = CHWC8;
  }

  Config config{};
  switch (variant) {
  case HWC: {
    config.invocC = m_hwc_configs[configKey].invocC;
    config.invocW = 1;
    config.invocH = m_hwc_configs[configKey].invocH;
    if (C < 256) {
      config.wgC = (C + config.invocC - 1) / config.invocC;
    } else {
      config.wgC = 128;
    }
    config.wgH = 1;
    config.wgW =
        (m_hwc_configs[configKey].wgSizeHint + config.wgC * config.wgH - 1) /
        (config.wgC * config.wgH);
    break;
  }
  case HWC8: {
    config.invocC = 8;
    config.invocW = m_hwc8_configs[configKey].invocW;
    config.invocH = m_hwc8_configs[configKey].invocH;
    if (C < 256) {
      config.wgC = (C + config.invocC - 1) / config.invocC;
    } else {
      config.wgC = 128;
    }
    config.wgH = 1;
    config.wgW =
        (m_hwc8_configs[configKey].wgSizeHint + config.wgC * config.wgH - 1) /
        (config.wgC * config.wgH);
    break;
  }
  case CHWC8: {
    config.invocC = 8;
    config.invocW = m_chwc8_configs[configKey].invocW;
    config.invocH = m_chwc8_configs[configKey].invocH;
    config.wgC = 1;
    config.wgH = 1;
    config.wgW = m_chwc8_configs[configKey].wgSize;
    break;
  }
  }

  auto shader = basic_activation_compile(
      m_compiler, m_srcPath, m_subgroupSize, in.format, out.format,
      static_cast<unsigned int>(in.channels.constant()), memory::Dtype::F16,
      acti.func, config);

  std::uint32_t tileC = config.invocC * config.wgC;
  std::uint32_t tileW = config.invocW * config.wgW;
  std::uint32_t tileH = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(in.channels, tileC, false, false);
  Sym workgroupCountY = symGraph.cdiv(in.width, tileW, false, false);
  Sym workgroupCountZ = symGraph.cdiv(in.height, tileH, false, false);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding("INPUT_SET", "INPUT_BINDING", Access::ReadOnly, inId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.width));
  dispatch.addPushConstant(PushConstant::Dynamic(in.height));
  dispatch.setSourcePath(m_srcPath);

  Sym reads =
      symGraph.mul(symGraph.mul(in.width, in.height), C * size_of(in.type));
  dispatch.setMemoryReads(reads);
  Sym writes =
      symGraph.mul(symGraph.mul(out.width, out.height), C * size_of(out.type));
  dispatch.setMemoryWrites(writes);

  dispatch.setName(name());
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", config.invocC,
      config.invocW, config.invocW, config.wgC, config.wgW, config.wgH));
  dispatch.usesCoopmat(false);

  switch (acti.func.kind()) {
  case ActivationFunctionKind::ReLU:
    dispatch.setOperation("relu(x)");
    // we do not count comparison as a FLOP!
    dispatch.setFlops(Sym::Const(0));
    break;
  case ActivationFunctionKind::LeakyReLU:
    dispatch.setOperation(
        fmt::format("leaky_relu(x,alpha={})", acti.func.leaky_relu().alpha));
    // leaky relu counts as one FLOP!
    dispatch.setFlops(symGraph.mul(symGraph.mul(out.width, out.height),
                                   symGraph.mul(out.channels, 1)));
    break;
  case ActivationFunctionKind::SiLU:
    dispatch.setOperation("sliu(x)");
    // silu counts as 20 FLOPs (obvious approximation because it contains a exp
    // which is kind oj)
    dispatch.setFlops(symGraph.mul(symGraph.mul(out.width, out.height),
                                   symGraph.mul(out.channels, 20)));
    break;
  case ActivationFunctionKind::Swish:
    dispatch.setOperation(
        fmt::format("swish(x,beta={})", acti.func.swish().beta));
    // silu counts as 20 FLOPs (obvious approximation because it contains a exp
    // which is kind oj)
    dispatch.setFlops(symGraph.mul(symGraph.mul(out.width, out.height),
                                   symGraph.mul(out.channels, 20)));
    break;
  }
}

memory::string BasicActivationShader::name() const {
  return "basic-activation";
}
} // namespace denox::compiler::shaders
