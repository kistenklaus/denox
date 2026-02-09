#include "denox/compiler/implement/shaders/pad/MemoryPadShader.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/dtype/dtype.hpp"

namespace denox::compiler::shaders {

MemoryPadShader::MemoryPadShader(spirv::GlslCompiler *compiler,
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
    if (tensor.format != TensorFormat::SSBO_HWC) {
      return false;
    }
    return true;
  };
  {
    Pattern p;
    auto in = p.matchNode();
    auto pad = in->matchOutgoing();
    auto out = pad->matchDst();

    pad->matchRank(1);
    pad->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::Pad; });

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    m_patternHandles.emplace_back(in, std::move(pad), out);
    m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                         std::move(out));
  }
}
memory::vector<unsigned int> MemoryPadShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];

  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  if (in.format != out.format) {
    return {};
  }
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
memory_pad_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                   TensorFormat inputFormat, TensorFormat outputFormat,
                   unsigned int channels,
                   const MemoryPadShader::Config &config) {
  auto shader = compiler->read(srcPath);
  shader.define("CH", channels);

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC && (channels % 8 == 0) &&
      config.invocC % 8 == 0) {
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
          "MemoryPadShader implements non vectorized layouts for format, "
          "which may be vectorized, this works, but is suboptimal!");
    }
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
    diag::unreachable();
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);
  return shader;
}

void MemoryPadShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &padId = match[patternHandles.pad];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const ComputeOpPad &pad = opGraph.get(padId).pad();

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
  auto shader = memory_pad_compile(m_compiler, m_srcPath, in.format, out.format,
                                   C, config);

  std::uint32_t tileX = config.invocC * config.wgC;
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX, false, false);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileY, false, false);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileZ, false, false);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding("INPUT_SET", "INPUT_BINDING", Access::ReadOnly, inId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);

  dispatch.addPushConstant( //
      PushConstant::Dynamic(out.width, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(out.height, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->left, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->right, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->top, memory::Dtype::U32));
  dispatch.addPushConstant( //
      PushConstant::Dynamic(pad->bottom, memory::Dtype::U32));

  dispatch.setName(name());
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H", config.invocC,
      config.invocW, config.invocH, config.wgC, config.wgW, config.wgH));
  dispatch.setOperation(fmt::format(
      "pad(x,({},{},{},{}),mode=replicate)",
      pad->left.isConstant() ? fmt::format("{}", pad->left.constant())
                             : "<dyn>",
      pad->right.isConstant() ? fmt::format("{}", pad->right.constant())
                              : "<dyn>",
      pad->top.isConstant() ? fmt::format("{}", pad->top.constant()) : "<dyn>",
      pad->bottom.isConstant() ? fmt::format("{}", pad->bottom.constant())
                               : "<dyn>"));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setFlops(Sym::Const(0));
  dispatch.usesCoopmat(false);
}
memory::string MemoryPadShader::name() const { return "memory-pad"; }
} // namespace denox::compiler::shaders
