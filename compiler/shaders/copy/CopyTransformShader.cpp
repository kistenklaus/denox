#include "shaders/copy/CopyTransformShader.hpp"
#include "diag/not_implemented.hpp"
#include "diag/unreachable.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/ComputeOp.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

enum class ConcatImplementationType {
  Explicit,
  Implicit,
  SingleCopy,
};

static constexpr unsigned int EXPLICIT_CONCAT_TAG = 0;
static constexpr unsigned int IMPLICIT_CONCAT_TAG = 1;

static constexpr unsigned int SINGLE_COPY_TAG = 2;

static std::vector<unsigned int> explicitConfigKeys{
    0x000000, 0x000100, 0x000200, 0x000300, 0x000400,

    0x010000, 0x010100, 0x010200, 0x010300, 0x010400,

    0x020000, 0x020100, 0x020200, 0x020300, 0x020400,

    0x030000, 0x030100, 0x030200, 0x030300, 0x030400,

    0x040000, 0x040100, 0x040200, 0x040300, 0x040400,
};

static std::vector<unsigned int> singleCopyConfigs{
    0x000002, 0x000102, 0x000202, 0x000302, 0x000402,
};

struct Config {
  unsigned int invocC;
  unsigned int invocW;
  unsigned int invocH;
  memory::optional<unsigned int> wgC;
  unsigned int wgW;
  unsigned int wgH;
};

static constexpr std::array<Config, 5> CONFIGS{
    Config{
        .invocC = 2,
        .invocW = 2,
        .invocH = 1,
        .wgC = 8,
        .wgW = 32,
        .wgH = 1,
    },
    Config{
        .invocC = 1,
        .invocW = 4,
        .invocH = 1,
        .wgC = memory::nullopt,
        .wgW = 32,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 4,
        .wgW = 64,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 2,
        .wgW = 128,
        .wgH = 1,
    },
    Config{
        .invocC = 8,
        .invocW = 1,
        .invocH = 1,
        .wgC = 1,
        .wgW = 256,
        .wgH = 1,
    },
};

CopyTransformShader::CopyTransformShader(GlslCompiler *compiler,
                                         const Options &options)
    : m_compiler(compiler),
      m_enableImplicitConcat(options.fusionRules.enableImplicitConcat) {

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::CHWC8 &&
        tensor.layout != memory::ActivationLayout::HWC) {
      return false;
    }
    return true;
  };
  {
    Pattern concatPattern;
    auto concat = concatPattern.matchEdge();
    concat->matchRank(2);
    auto in0 = concat->matchSrc(0);
    auto in1 = concat->matchSrc(1);
    auto out = concat->matchDst();
    in0->matchValue(supportedTensor);
    in1->matchValue(supportedTensor);
    concat->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Concat; });
    m_patternHandles.emplace_back(in0, in1, out);
    m_capabilities.patterns.emplace_back(std::move(concatPattern),
                                         std::move(in0), std::move(in1),
                                         std::move(out));
  }
}

static ConcatImplementationType
tryImplicitConcat(memory::ActivationLayout src0Layout,
                  memory::ActivationLayout src1Layout,
                  memory::ActivationLayout dstLayout, memory::Dtype src0Type,
                  memory::Dtype src1Type, memory::Dtype dstType,
                  const CanoModel::Graph::NodeHandle &src0,
                  const CanoModel::Graph::NodeHandle &src1) {

  if (src0Layout != memory::ActivationLayout::CHWC8 ||
      src1Layout != memory::ActivationLayout::CHWC8 ||
      dstLayout != memory::ActivationLayout::CHWC8) {
    return ConcatImplementationType::Explicit;
  }
  if (!(src0Type == src1Type && src1Type == dstType)) {
    return ConcatImplementationType::Explicit;
  }
  size_t src0ConcatFanout = 0;
  for (const auto &e : src0->outgoing()) {
    if (e.value().tag() == ComputeOpTag::Concat) {
      ++src0ConcatFanout;
    }
  }

  size_t src1ConcatFanout = 0;
  for (const auto &e : src1->outgoing()) {
    if (e.value().tag() == ComputeOpTag::Concat) {
      ++src1ConcatFanout;
    }
  }
  if (src0ConcatFanout > 1 && src1ConcatFanout > 1) {
    return ConcatImplementationType::Explicit;
  }

  if (src0ConcatFanout == 1 && src1ConcatFanout == 1) {
    return ConcatImplementationType::Implicit;
  }
  // see #8246379 for a unstable implementation of lifetime based implicit
  // concat.

  return ConcatImplementationType::Explicit;
}

memory::vector<unsigned int> CopyTransformShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
    [[maybe_unused]] unsigned int patternEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  unsigned int pattern = patternEnc & PATTERN_MASK;
  [[maybe_unused]] unsigned int mode = patternEnc & CONCAT_MODE_MASK;
  assert(mode == 0);

  const auto &patternHandles = m_patternHandles[pattern];
  const memory::NodeId src0Id = match[patternHandles.src0];
  const memory::NodeId src1Id = match[patternHandles.src1];
  const memory::NodeId dstId = match[patternHandles.dst];

  const TensorInstance &src0 = graph.get(src0Id);
  const TensorInstance &src1 = graph.get(src1Id);
  const TensorInstance &dst = graph.get(dstId);

  memory::ActivationLayout src0Layout =
      memory::ActivationLayout::demote(src0.layout, src0.channels);
  memory::ActivationLayout src1Layout =
      memory::ActivationLayout::demote(src1.layout, src1.channels);
  memory::ActivationLayout dstLayout =
      memory::ActivationLayout::demote(dst.layout, dst.channels);

  if (!(src0Layout == src1Layout && src1Layout == dstLayout)) {
    return {};
  }

  ConcatImplementationType implementationType =
      tryImplicitConcat(src0Layout, src1Layout, dstLayout, src0.type, src1.type,
                        dst.type, src0.originalNode, src1.originalNode);

  switch (implementationType) {
  case ConcatImplementationType::Explicit:
    return explicitConfigKeys;
  case ConcatImplementationType::Implicit:
    return {IMPLICIT_CONCAT_TAG};
  case ConcatImplementationType::SingleCopy:
    throw std::runtime_error("Not implemented HERE");
    return singleCopyConfigs;
  }
  diag::unreachable();
}

float CopyTransformShader::speedup(unsigned int config) const {
  if (config & IMPLICIT_CONCAT_TAG) {
    return 0;
  } else {
    return 2;
  }
}

static GlslCompilerInstance
compile(GlslCompiler *compiler, const io::Path &srcPath,
        unsigned int inputChannelOffset, unsigned int inputChannels,
        unsigned int outputChannelOffset, unsigned int outputChannels,
        memory::ActivationLayout inputLayout,
        memory::ActivationLayout outputLayout, const Config &config) {
  auto shader = compiler->read(srcPath);

  shader.define("IN_CH_OFFSET", inputChannelOffset);
  shader.define("IN_CH", inputChannels);
  shader.define("OUT_CH_OFFSET", outputChannelOffset);
  shader.define("OUT_CH", outputChannels);

  if (inputLayout == memory::ActivationLayout::HWC &&
      outputLayout == memory::ActivationLayout::HWC &&
      (inputChannels % 8 != 0 || outputChannels % 8 != 0)) {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");
  } else if (inputLayout == memory::ActivationLayout::HWC &&
             outputLayout == memory::ActivationLayout::HWC &&
             (inputChannels % 8 == 0 && outputChannels % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");
  } else if (inputLayout == memory::ActivationLayout::CHWC8 &&
             outputLayout == memory::ActivationLayout::CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
  } else {
    compiler::diag::unreachable();
  }

  shader.define("INVOC_C", config.invocC);
  shader.define("INVOC_W", config.invocW);
  shader.define("INVOC_H", config.invocH);
  shader.define("WG_C", config.wgC.value_or(inputChannels));
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);
  return shader;
}

void CopyTransformShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  static_assert(sizeof(unsigned int) == 4);
  uint8_t type = configEnc & 0xFF;

  const auto &patternHandles = m_patternHandles[pattern];

  memory::NodeId src0Id = match[patternHandles.src0];
  memory::NodeId src1Id = match[patternHandles.src1];
  memory::NodeId dstId = match[patternHandles.dst];

  if (type == IMPLICIT_CONCAT_TAG) {
    impl.createImplicitConcatConstrain(src0Id, src1Id, dstId);
    return;
  }

  if (type == SINGLE_COPY_TAG) {
    diag::not_implemented();
    return;
  } else if (type == EXPLICIT_CONCAT_TAG) {

    const auto &dst = opGraph.get(dstId);

    {
      const auto &src0 = opGraph.get(src0Id);
      uint8_t config0Key = (configEnc >> 8) & 0xFF;
      const Config &config0 = CONFIGS[config0Key];
      auto shader0 = compile(m_compiler, m_srcPath, 0, src0.channels, 0,
                             dst.channels, src0.layout, dst.layout, config0);

      std::uint32_t tileX = config0.invocC * config0.wgC.value_or(src0.channels);
      std::uint32_t tileY =
          config0.invocW * config0.wgW;
      std::uint32_t tileZ = config0.invocH * config0.wgH;

      Sym workgroupCountX = symGraph.cdiv(src0.channels, tileX);
      Sym workgroupCountY = symGraph.cdiv(src0.extent.x.asSym(), tileY);
      Sym workgroupCountZ = symGraph.cdiv(src0.extent.y.asSym(), tileZ);

      auto copySrc0Dispatch =
          impl.registerDispatch(std::move(shader0), workgroupCountX,
                                workgroupCountY, workgroupCountZ);
      copySrc0Dispatch.addBinding(0, 0, AccessFlag::ReadOnly, src0Id);
      copySrc0Dispatch.addBinding(0, 1, AccessFlag::WriteOnly, dstId);
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.x));
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.y));
      copySrc0Dispatch.setName("explicit-concat-copy-src0");
      copySrc0Dispatch.setSourcePath(m_srcPath);

      Sym reads = symGraph.mul(src0.extent.x.asSym(), src0.extent.y.asSym(),
                               src0.channels * src0.type.size());
      Sym writes = symGraph.mul(src0.extent.x.asSym(), src0.extent.y.asSym(),
                                src0.channels * dst.type.size());
      copySrc0Dispatch.setMemoryReads(reads);
      copySrc0Dispatch.setMemoryWrites(writes);
      copySrc0Dispatch.setDebugInfo(fmt::format("CopyTransformShader\n"
                                                "- IN_LAYOUT:  {}\n"
                                                "- OUT_LAYOUT: {}\n",
                                                src0.layout.to_string(),
                                                dst.layout.to_string()));

      copySrc0Dispatch.setInputDesc(
          fmt::format("{}[{}]", src0.layout.to_string(), src0.channels));
      copySrc0Dispatch.setOutputDesc(
          fmt::format("{}[{}]", dst.layout.to_string(), dst.channels));
    }

    {
      const auto &src1 = opGraph.get(src1Id);
      uint8_t config1Key = (configEnc >> 16) & 0xFF;
      const Config &config1 = CONFIGS[config1Key];
      auto shader1 =
          compile(m_compiler, m_srcPath, 0, src1.channels, src1.channels,
                  dst.channels, src1.layout, dst.layout, config1);

      std::uint32_t tileX = config1.invocC * config1.wgC.value_or(src1.channels);
      std::uint32_t tileY =
          config1.invocW * config1.wgW;
      std::uint32_t tileZ = config1.invocH * config1.wgH;

      Sym workgroupCountX = symGraph.cdiv(src1.channels, tileX);
      Sym workgroupCountY = symGraph.cdiv(src1.extent.x.asSym(), tileY);
      Sym workgroupCountZ = symGraph.cdiv(src1.extent.y.asSym(), tileZ);

      auto copySrc1Dispatch =
          impl.registerDispatch(std::move(shader1), workgroupCountX,
                                workgroupCountY, workgroupCountZ);
      copySrc1Dispatch.addBinding(0, 0, AccessFlag::ReadOnly, src1Id);
      copySrc1Dispatch.addBinding(0, 1, AccessFlag::WriteOnly, dstId);
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.x));
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.y));
      copySrc1Dispatch.setName("explicit-concat-copy-src1");
      copySrc1Dispatch.setSourcePath(m_srcPath);

      Sym reads = symGraph.mul(src1.extent.x.asSym(), src1.extent.y.asSym(),
                               src1.channels * src1.type.size());
      Sym writes = symGraph.mul(src1.extent.x.asSym(), src1.extent.y.asSym(),
                                src1.channels * dst.type.size());
      copySrc1Dispatch.setMemoryReads(reads);
      copySrc1Dispatch.setMemoryWrites(writes);
      copySrc1Dispatch.setDebugInfo(fmt::format("CopyTransformShader\n"
                                                "- IN_LAYOUT:  {}\n"
                                                "- OUT_LAYOUT: {}\n",
                                                src1.layout.to_string(),
                                                dst.layout.to_string()));

      copySrc1Dispatch.setInputDesc(
          fmt::format("{}[{}]", src1.layout.to_string(), src1.channels));
      copySrc1Dispatch.setOutputDesc(
          fmt::format("{}[{}]", dst.layout.to_string(), dst.channels));
    }
  }
}

memory::string CopyTransformShader::name(unsigned int,
                                         unsigned int config) const {
  if (config & IMPLICIT_CONCAT_TAG) {
    return "implicit-concat";
  } else if (config & SINGLE_COPY_TAG) {
    return "single-copy-concat";
  } else {
    return "explicit-concat";
  }
}

} // namespace denox::compiler::shaders
