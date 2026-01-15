#include "denox/compiler/implement/shaders/slice/MemorySliceShader.hpp"
#include "denox/diag/invalid_state.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

struct MemorySliceConfig {
  unsigned int invocC;
  unsigned int invocW;
  unsigned int invocH;
  memory::optional<unsigned int> wgC;
  unsigned int wgW;
  unsigned int wgH;
};

static std::array<MemorySliceConfig, 5> MEMORY_SLICE_CONFIGS{
    MemorySliceConfig{
        .invocC = 2,
        .invocW = 2,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    MemorySliceConfig{
        .invocC = 1,
        .invocW = 4,
        .invocH = 1,
        .wgC = memory::nullopt,
        .wgW = 32,
        .wgH = 1,
    },
    MemorySliceConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    MemorySliceConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    MemorySliceConfig{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },
};

MemorySliceShader::MemorySliceShader(spirv::GlslCompiler *compiler)
    : m_compiler(compiler) {

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
    if (tensor.format != TensorFormat::SSBO_HWC) {
      return false;
    }
    return true;
  };
  {
    Pattern p;
    auto in = p.matchNode();
    auto slice = in->matchOutgoing();
    auto out = slice->matchDst();
    slice->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::Slice; });

    in->matchValue(supportedTensor);
    out->matchValue(supportedTensor);

    m_patternHandles.emplace_back(in, std::move(slice), out);
    m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                         std::move(out));
  }
}
memory::vector<unsigned int> MemorySliceShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  if (in.format != out.format) {
    return {};
  }

  uint32_t cblocksize;
  switch (in.format) {
  case TensorFormat::SSBO_HWC:
    if (in.channels.constant() % 8 == 0) {
      cblocksize = 8;
    } else {
      cblocksize = 1;
    }
    break;
  case TensorFormat::SSBO_CHW:
    cblocksize = 1;
    break;
  case TensorFormat::SSBO_CHWC8:
    cblocksize = 8;
    break;
  case TensorFormat::Optimal:
  case TensorFormat::TEX_RGBA:
  case TensorFormat::TEX_RGB:
  case TensorFormat::TEX_RG:
  case TensorFormat::TEX_R:
    diag::invalid_state();
    break;
  }

  memory::vector<unsigned int> configs;
  configs.reserve(MEMORY_SLICE_CONFIGS.size());
  for (unsigned int c = 0; c < MEMORY_SLICE_CONFIGS.size(); ++c) {
    if (MEMORY_SLICE_CONFIGS[c].invocC % cblocksize == 0) {
      configs.push_back(c);
    }
  }
  return configs;
}

static spirv::GlslCompilerInstance
memory_slice_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
        TensorFormat inputFormat, TensorFormat outputFormat,
        unsigned int channels, const MemorySliceConfig &config) {
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
          "MemorySliceShader implements non vectorized layouts for format, "
          "which may be vectorized, this works, but is suboptimal!");
    }
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);

    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
  } else {
    throw std::runtime_error("not supported");
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC.value_or(channels));
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);
  return shader;
}

void MemorySliceShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {

  const MemorySliceConfig &config = MEMORY_SLICE_CONFIGS[configKey];

  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  memory::EdgeId opId = match[patternHandle.slice];

  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &slice = opGraph.get(opId).slice();

  assert(in.channels == out.channels);

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  auto shader =
      memory_slice_compile(m_compiler, m_srcPath, in.format, out.format, C, config);

  std::uint32_t tileX = config.invocC * config.wgC.value_or(C);
  std::uint32_t tileY = config.invocW * config.wgW;
  std::uint32_t tileZ = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileX);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileY);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileZ);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, Access::ReadOnly, inId);
  dispatch.addBinding(0, 1, Access::WriteOnly, outId);

  dispatch.addPushConstant(PushConstant::Dynamic(slice->top));
  dispatch.addPushConstant(PushConstant::Dynamic(slice->left));
  dispatch.addPushConstant(PushConstant::Dynamic(in.width));
  dispatch.addPushConstant(PushConstant::Dynamic(in.height));
  dispatch.addPushConstant(PushConstant::Dynamic(out.width));
  dispatch.addPushConstant(PushConstant::Dynamic(out.height));
  dispatch.setName(name(pattern, 0));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setDebugInfo(fmt::format("MemorySliceShader\n"
                                    "- IN_LAYOUT:  {}\n"
                                    "- OUT_LAYOUT: {}\n",
                                    in.format, out.format));
}

memory::string MemorySliceShader::name(unsigned int, unsigned int) const {
  return "memory-slice";
}
} // namespace denox::compiler::shaders
