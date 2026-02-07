#include "denox/compiler/implement/shaders/upsample/BasicUpsampleShader.hpp"
#include "denox/common/FilterMode.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include <cassert>
#include <exception>

namespace denox::compiler::shaders {

// static std::array<BasicUpsampleConfig, 5> BASIC_UPSAMPLE_CONFIGS{
//     BasicUpsampleConfig{
//         // not supported
//         .invocC = 2,
//         .invocW = 2,
//         .invocH = 1,
//         .wgC = 8,
//         .wgW = 32,
//         .wgH = 1,
//     },
//     BasicUpsampleConfig{
//         // not supported
//         .invocC = 1,
//         .invocW = 4,
//         .invocH = 1,
//         .wgC = memory::nullopt,
//         .wgW = 32,
//         .wgH = 1,
//     },
//     BasicUpsampleConfig{
//         // 0.383ms
//         .invocC = 8,
//         .invocW = 1,
//         .invocH = 1,
//         .wgC = 4,
//         .wgW = 64,
//         .wgH = 1,
//     },
//     BasicUpsampleConfig{
//         // 0.319ms
//         .invocC = 8,
//         .invocW = 1,
//         .invocH = 1,
//         .wgC = 2,
//         .wgW = 128,
//         .wgH = 1,
//     },
//     BasicUpsampleConfig{
//         // 0.318ms
//         .invocC = 8,
//         .invocW = 1,
//         .invocH = 1,
//         .wgC = 1,
//         .wgW = 256,
//         .wgH = 1,
//     },
//
// };

BasicUpsampleShader::BasicUpsampleShader(spirv::GlslCompiler *compiler,
                                         const CompileOptions &options)
    : m_compiler(compiler),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize) {

  { // ===== Create configuration space
    // NOTE: We stick with enumerating a bunch of reasonable configs here,
    // no need to do to generate all possible configurations

    // NOTE: This is a stupidly large search space for such a simple op!
    {
      for (uint32_t wg_C = 1; wg_C <= 256; wg_C += 1) {
        for (uint32_t wg_W = 1; wg_W <= 256; wg_W <<= 2) {
          for (uint32_t wg_H = 1; wg_H <= 256; wg_H <<= 2) {

            if (options.deviceInfo.limits.maxComputeWorkGroupSize[0] < wg_C) {
              continue;
            }
            if (options.deviceInfo.limits.maxComputeWorkGroupSize[1] < wg_W) {
              continue;
            }
            if (options.deviceInfo.limits.maxComputeWorkGroupSize[2] < wg_H) {
              continue;
            }

            uint32_t wgSize = wg_C * wg_W * wg_H;
            if ((wgSize < options.deviceInfo.subgroup.subgroupSize) ||
                (wgSize >
                 std::min(
                     options.deviceInfo.limits.maxComputeWorkGroupInvocations,
                     512u))) {
              continue;
            }
            if ((wgSize % options.deviceInfo.subgroup.subgroupSize) != 0) {
              continue;
            }

            for (uint32_t invoc_C = 1; invoc_C <= 8; invoc_C++) {
              for (uint32_t invoc_W = 1; invoc_W <= 8; invoc_W++) {
                for (uint32_t invoc_H = 1; invoc_H <= 8; invoc_H++) {
                  if (invoc_C * invoc_W * invoc_H > 64) {
                    continue;
                  }

                  // fmt::println("INVOC_C={},WG_C={}", invoc_C, wg_C);

                  m_configs.push_back(BasicUpsampleConfig{
                      .invocC = invoc_C,
                      .invocW = invoc_W,
                      .invocH = invoc_H,
                      .wgC = wg_C,
                      .wgW = wg_W,
                      .wgH = wg_H,
                  });
                }
              }
            }
          }
        }
      }
    }
  }
  fmt::println("upsample-config-space: {}", m_configs.size());

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != TensorDataType::Float16) {
      return false;
    }
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    if (tensor.storage != TensorStorage::StorageBuffer) {
      return false;
    }

    return tensor.format == TensorFormat::SSBO_HWC ||
           tensor.format == TensorFormat::SSBO_CHWC8;
  };
  {
    Pattern upsamplePattern;
    auto in = upsamplePattern.matchNode();
    auto upsample = in->matchOutgoing();
    auto out = upsample->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    upsample->matchRank(1);
    upsample->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Upsample) {
        return false;
      }
      const auto &upsample = op.upsample();
      if (upsample.mode != FilterMode::Nearest) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(upsample), out);
    m_capabilities.patterns.emplace_back(std::move(upsamplePattern),
                                         std::move(in), std::move(out));
  }
}
memory::vector<unsigned int> BasicUpsampleShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);

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

  memory::vector<unsigned int> promissing;

  const bool vectorized = C % 8 == 0;
  fmt::println("C = {}", C);

  for (unsigned int c = 0; c < m_configs.size(); ++c) {
    const auto &config = m_configs[c];

    uint32_t ctile = config.invocC * config.wgC;

    uint32_t cdispatches = (C + ctile - 1) / ctile;
    // fmt::println("c-dispatches: {}", cdispatches);
    if (C <= 256 && cdispatches != 1) {
      continue;
    }

    if (vectorized) {
      if (config.invocC % 8 != 0) {
        continue; // <- invalid configconfig
      }

      if (C % ctile != 0) {
        continue;
      }

    } else {
      if (config.invocC > 2) {
        continue;
      }
    }

    // fmt::println("{}x{}x{}  {}x{}x{}", config.invocC, config.invocW,
    //              config.invocH, config.wgC, config.wgW, config.wgH);

    promissing.push_back(c);
  }
  fmt::println("config-space: {}", promissing.size());
  return promissing;
}

static spirv::GlslCompilerInstance
basic_upsample_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                       TensorFormat inputFormat, TensorFormat outputFormat,
                       unsigned int channels, unsigned int scalingFactor,
                       const BasicUpsampleConfig &config) {
  auto shader = compiler->read(srcPath);
  shader.enableDenoxPreprocessor();

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC &&
      (channels % 8 == 0 && config.invocC % 8 == 0)) {
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
          "BasicUpsampleShader implements non vectorized layouts for format, "
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
    throw std::logic_error("Invalid state");
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  shader.define("CH", channels);
  shader.define("SCALING_FACTOR", scalingFactor);
  return shader;
}

void BasicUpsampleShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match,
    SymGraph &symGraph) const {
  const BasicUpsampleConfig config = m_configs[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId upsampleId = match[patternHandles.upsample];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &upsample = opGraph.get(upsampleId).upsample();

  uint32_t C = static_cast<uint32_t>(in.channels.constant());
  assert(C == out.channels.constant());

  auto shader =
      basic_upsample_compile(m_compiler, m_srcPath, in.format, out.format, C,
                             upsample.scalingFactor, config);

  std::uint32_t tileX = config.invocC * config.wgC;
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding("INPUT_SET", "INPUT_BINDING", Access::ReadOnly, inId);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.width));
  dispatch.addPushConstant(PushConstant::Dynamic(in.height));
  dispatch.setName(name());
  dispatch.setOperation(fmt::format("upsample(x,scale_factor={},mode=nearest)",
                                    upsample.scalingFactor));
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", config.invocC,
      config.invocW, config.invocH, config.wgC, config.wgW, config.wgH));

  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));

  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setFlops(Sym::Const(0));
  dispatch.usesCoopmat(false);
}
memory::string BasicUpsampleShader::name() const { return "basic-upsample"; }
} // namespace denox::compiler::shaders
