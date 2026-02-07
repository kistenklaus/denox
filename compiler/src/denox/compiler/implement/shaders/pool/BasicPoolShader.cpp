#include "denox/compiler/implement/shaders/pool/BasicPoolShader.hpp"
#include "denox/common/PoolFunction.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/logging.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

BasicPoolShader::BasicPoolShader(spirv::GlslCompiler *compiler,
                                 const CompileOptions &options)
    : m_compiler(compiler) {

  { // ========= Generate config space ============
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

                m_configs.push_back(BasicPoolConfig{
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

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != TensorDataType::Float16) {
      return false;
    }
    if (tensor.storage != TensorStorage::StorageBuffer) {
      return false;
    }
    if (tensor.channels.isSymbolic()) {
      return false;
    }
    return tensor.format == TensorFormat::SSBO_HWC ||
           tensor.format == TensorFormat::SSBO_CHWC8;
  };

  {

    Pattern poolPattern;
    auto in = poolPattern.matchNode();
    auto pool = in->matchOutgoing();
    auto out = pool->matchDst();

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    pool->matchRank(1);

    pool->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Pool) {
        return false;
      }
      const auto &pool = op.pool();
      if (pool->func != PoolFunction::Max) {
        return false;
      }
      if (pool->stride != pool->kernelSize) {
        return false;
      }
      if (pool->padding != memory::uvec2(0, 0)) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(pool), out);
    m_capabilities.patterns.emplace_back(std::move(poolPattern), std::move(in),
                                         std::move(out));
  }
}
memory::vector<unsigned int> BasicPoolShader::acceptMatch(
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
    return {};
  }
  if (out.type != TensorDataType::Float16) {
    return {};
  }

  // uint32_t cblocksize;
  // switch (in.format) {
  // case TensorFormat::SSBO_HWC:
  //   if (in.channels.constant() % 8 == 0) {
  //     cblocksize = 8;
  //   } else {
  //     cblocksize = 1;
  //   }
  //   break;
  // case TensorFormat::SSBO_CHW:
  //   cblocksize = 1;
  //   break;
  // case TensorFormat::SSBO_CHWC8:
  //   cblocksize = 8;
  //   break;
  // case TensorFormat::Optimal:
  // case TensorFormat::TEX_RGBA:
  // case TensorFormat::TEX_RGB:
  // case TensorFormat::TEX_RG:
  // case TensorFormat::TEX_R:
  //   diag::invalid_state();
  //   break;
  // }

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  memory::vector<unsigned int> promissing;
  for (unsigned int c = 0; c < m_configs.size(); ++c) {
    const auto &config = m_configs[c];

    uint32_t ctile = config.invocC * config.wgC;
    uint32_t cdispatches = (C + ctile - 1) / ctile;
    if (C <= 256 && cdispatches != 1) {
      continue;
    }

    const bool vec = C % 8 == 0;
    if (vec) {
      if (config.invocC % 8 != 0) {
        continue; // invalid config.
      }
      if (C % ctile != 0) {
        continue;
      }
    } else {
      if (config.invocC > 2) {
        continue;
        continue;
      }
    }

    promissing.push_back(c);
    // unsigned int invocC = config.invocC;
    // if (invocC % cblocksize == 0) {
    // }
  }
  // fmt::println("max-pool-config-space: {}", promissing.size());
  return promissing;
}

static spirv::GlslCompilerInstance
basic_pool_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                   TensorFormat inputFormat, TensorFormat outputFormat,
                   unsigned int channels, memory::uvec2 kernelSize,
                   memory::uvec2 stride, memory::uvec2 padding,
                   const BasicPoolConfig &config) {
  auto shader = compiler->read(srcPath);

  uint32_t invocC = config.invocC;
  // if (config.invocC) {
  //   invocC = *config.invocC;
  // } else {
  //   const unsigned int ix = (channels + config.cdiv - 1) / config.cdiv;
  //   invocC = ix;
  // }

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC && (channels % 8 == 0) &&
      (invocC % 8 == 0)) {
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
          "BasicPoolShader implements non vectorized layouts for format, "
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

  shader.define("INVOC_C", invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);

  shader.define("CH", channels);

  shader.define("KERNEL_X", kernelSize.x);
  shader.define("KERNEL_Y", kernelSize.y);
  shader.define("STRIDE_X", stride.x);
  shader.define("STRIDE_Y", stride.y);
  shader.define("PADDING_X", padding.x);
  shader.define("PADDING_Y", padding.y);
  return shader;
}

void BasicPoolShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const BasicPoolConfig &config = m_configs[configKey];

  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId poolId = match[patternHandles.pool];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &pool = opGraph.get(poolId).pool();

  assert(in.channels == out.channels);

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  auto shader =
      basic_pool_compile(m_compiler, m_srcPath, in.format, out.format, C,
                         pool->kernelSize, pool->stride, pool->padding, config);

  unsigned int invocC = config.invocC;
  // if (config.invocC) {
  //   invocC = *config.invocC;
  // } else {
  //   invocC = (C + config.cdiv - 1) / config.cdiv;
  // }

  std::uint32_t tileX = invocC * config.wgC;
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
  dispatch.setOperation(
      fmt::format("max_pool2d(x,kernel_size=({},{}),stride=({"
                  "},{}),padding=({},{}),dialation=1,ceil_mode=false)",
                  pool->kernelSize.x, pool->kernelSize.y, pool->stride.x,
                  pool->stride.y, pool->padding.x, pool->padding.y));
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", invocC,
      config.invocW, config.invocH, config.wgC, config.wgW, config.wgH));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setFlops(Sym::Const(0));
  dispatch.usesCoopmat(false);
}
memory::string BasicPoolShader::name() const { return "basic-pool"; }
} // namespace denox::compiler::shaders
