#include "denox/compiler/implement/shaders/copy/CopyTransformShader.hpp"
#include "denox/common/ComputeOp.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
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

CopyTransformShader::CopyTransformShader(spirv::GlslCompiler *compiler,
                                         const CompileOptions &options)
    : m_compiler(compiler),
      m_enableImplicitConcat(options.features.enableImplicitConcat),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize) {

  {
    auto fd = io::File::open(io::Path::assets() /
                                 "compiler/src/denox/compiler/implement/"
                                 "shaders/copy/copy_transform.configs",
                             io::File::OpenMode::Read);
    std::string str;
    str.resize(fd.size());
    fd.read_exact(std::span<std::byte>(
        reinterpret_cast<std::byte *>(str.data()), str.size()));
    std::stringstream ss(str);

    Config config;
    while (ss >> config.invocC >> config.invocW >> config.invocH >>
           config.wgC >> config.wgW >> config.wgH) {

      if (config.wgC > options.deviceInfo.limits.maxComputeWorkGroupSize[0]) {
        continue;
      }
      if (config.wgW > options.deviceInfo.limits.maxComputeWorkGroupSize[1]) {
        continue;
      }
      if (config.wgH > options.deviceInfo.limits.maxComputeWorkGroupSize[2]) {
        continue;
      }
      uint32_t wg_size = config.wgC * config.wgH * config.wgW;
      if (wg_size > options.deviceInfo.limits.maxComputeWorkGroupInvocations) {
        continue;
      }
      m_configs.push_back(config);
    }
    if (m_configs.size() > 256) {
      throw std::runtime_error(
          "Too many copy-transform configs for 8-bit encoding");
    }
  }

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
    if (tensor.format != TensorFormat::SSBO_CHWC8 &&
        tensor.format != TensorFormat::SSBO_HWC) {
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
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::Concat; });
    m_patternHandles.emplace_back(in0, in1, out);
    m_capabilities.patterns.emplace_back(std::move(concatPattern),
                                         std::move(in0), std::move(in1),
                                         std::move(out));
  }
}

static ConcatImplementationType
tryImplicitConcat(TensorFormat src0Layout, TensorFormat src1Layout,
                  TensorFormat dstLayout, TensorDataType src0Type,
                  TensorDataType src1Type, TensorDataType dstType,
                  const CanoModel::Graph::NodeHandle &src0,
                  const CanoModel::Graph::NodeHandle &src1) {

  if (src0Layout != TensorFormat::SSBO_CHWC8 ||
      src1Layout != TensorFormat::SSBO_CHWC8 ||
      dstLayout != TensorFormat::SSBO_CHWC8) {
    return ConcatImplementationType::Explicit;
  }
  if (!(src0Type == src1Type && src1Type == dstType)) {
    return ConcatImplementationType::Explicit;
  }
  size_t src0ConcatFanout = 0;
  for (const auto &e : src0->outgoing()) {
    if (e.value().tag() == ComputeOpKind::Concat) {
      ++src0ConcatFanout;
    }
  }

  size_t src1ConcatFanout = 0;
  for (const auto &e : src1->outgoing()) {
    if (e.value().tag() == ComputeOpKind::Concat) {
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

  TensorFormat src0Format = src0.format;
  TensorFormat src1Format = src1.format;
  TensorFormat dstFormat = dst.format;

  if (!(src0Format == src1Format && src1Format == dstFormat)) {
    return {};
  }

  ConcatImplementationType implementationType =
      ConcatImplementationType::Explicit;
  if (m_enableImplicitConcat) {
    implementationType = tryImplicitConcat(
        src0Format, src1Format, dstFormat, src0.type, src1.type, dst.type,
        src0.originalNode, src1.originalNode);
  }

  auto supported = [&](uint32_t wgC, uint32_t wgW, uint32_t wgH,
                       uint32_t invocC, TensorFormat format,
                       unsigned int channels) -> bool {
    uint32_t workgroupInvocationCount = wgC * wgW * wgH;
    if (workgroupInvocationCount >= m_maxComputeWorkGroupInvocations) {
      return false;
    }
    if (wgC >= m_maxComputeWorkGroupSize[0]) {
      return false;
    }
    if (wgW >= m_maxComputeWorkGroupSize[1]) {
      return false;
    }
    if (wgH >= m_maxComputeWorkGroupSize[2]) {
      return false;
    }
    uint32_t cblocksize;
    switch (format) {
    case TensorFormat::SSBO_HWC:
      if (channels % 8 == 0) {
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
    if (invocC % cblocksize != 0) {
      return false;
    }
    return true;
  };

  std::vector<unsigned int> configs;
  switch (implementationType) {
  case ConcatImplementationType::Explicit:
    configs.reserve(m_configs.size() * m_configs.size());
    for (unsigned int c0 = 0; c0 < m_configs.size(); ++c0) {
      const auto &config0 = m_configs[c0];
      if (!supported(config0.wgC, config0.wgW, config0.wgH, config0.invocC,
                     src0Format,
                     static_cast<uint32_t>(src0.channels.constant()))) {
        continue;
      }
      for (unsigned int c1 = 0; c1 < m_configs.size(); ++c1) {
        const auto &config1 = m_configs[c1];
        if (supported(config1.wgC, config1.wgW, config1.wgH, config1.invocC,
                      src1Format,
                      static_cast<uint32_t>(src1.channels.constant()))) {
          configs.push_back((c1 << 16) | (c0 << 8) | EXPLICIT_CONCAT_TAG);
        }
      }
    }
    break;
  case ConcatImplementationType::Implicit:
    return {IMPLICIT_CONCAT_TAG};
  case ConcatImplementationType::SingleCopy:
    throw std::runtime_error("Not implemented");
  }
  return configs;
}

float CopyTransformShader::speedup(unsigned int config) const {
  if (config & IMPLICIT_CONCAT_TAG) {
    return 0;
  } else {
    return 2;
  }
}

static spirv::GlslCompilerInstance copy_transform_compile(
    spirv::GlslCompiler *compiler, const io::Path &srcPath,
    unsigned int inputChannelOffset, unsigned int inputChannels,
    unsigned int outputChannelOffset, unsigned int outputChannels,
    TensorFormat inputFormat, TensorFormat outputFormat,
    bool allowVectorization, const CopyTransformShader::Config &config) {
  auto shader = compiler->read(srcPath);

  shader.define("IN_CH_OFFSET", inputChannelOffset);
  shader.define("IN_CH", inputChannels);
  shader.define("OUT_CH_OFFSET", outputChannelOffset);
  shader.define("OUT_CH", outputChannels);

  if (inputFormat == TensorFormat::SSBO_HWC &&
      outputFormat == TensorFormat::SSBO_HWC &&
      (inputChannels % 8 == 0 && outputChannels % 8 == 0) &&
      (config.invocC % 8 == 0) && allowVectorization) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");
  } else if (inputFormat == TensorFormat::SSBO_HWC &&
             outputFormat == TensorFormat::SSBO_HWC) {
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

void CopyTransformShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  static_assert(sizeof(decltype(configEnc)) == 4);
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

    const auto &src0 = opGraph.get(src0Id);

    uint32_t src0Channels = static_cast<uint32_t>(src0.channels.constant());
    uint32_t dstChannels = static_cast<uint32_t>(dst.channels.constant());
    {
      uint8_t config0Key = (configEnc >> 8) & 0xFF;
      const Config &config0 = m_configs[config0Key];
      auto shader0 = copy_transform_compile(
          m_compiler, m_srcPath, 0, src0Channels, 0, dstChannels, src0.format,
          dst.format, true, config0);

      std::uint32_t tileX = config0.invocC * config0.wgC;
      std::uint32_t tileY = config0.invocW * config0.wgW;
      std::uint32_t tileZ = config0.invocH * config0.wgH;

      // tileX = 2 * 8 = 16
      // tileY = 2 * 32 = 64
      // tileZ = 1 * 1 = 1

      Sym workgroupCountX =
          symGraph.cdiv(src0.channels, tileX, false, false); // 8 / 16 = 1
      Sym workgroupCountY =
          symGraph.cdiv(src0.width, tileY, false, false); // 1920 / 64 = 30
      Sym workgroupCountZ =
          symGraph.cdiv(src0.height, tileZ); // 1080 / 1 = 1080

      auto copySrc0Dispatch =
          impl.registerDispatch(std::move(shader0), workgroupCountX,
                                workgroupCountY, workgroupCountZ);
      copySrc0Dispatch.addBinding("SRC_SET", "SRC_BINDING", Access::ReadOnly,
                                  src0Id);
      copySrc0Dispatch.addBinding("DST_SET", "DST_BINDING", Access::WriteOnly,
                                  dstId);
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.width));
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.height));

      Sym reads = symGraph.mul(src0.width, src0.height,
                               src0Channels * size_of(src0.type));
      Sym writes = symGraph.mul(src0.width, src0.height,
                                src0Channels * size_of(dst.type));
      copySrc0Dispatch.setMemoryReads(reads);
      copySrc0Dispatch.setMemoryWrites(writes);

      copySrc0Dispatch.setFlops(Sym::Const(0));
      // copySrc0Dispatch.usesCoopmat(false);

      // r[0:3,:,:]=x
      copySrc0Dispatch.setOperation(
          fmt::format("r[{}:{},:,:]=x", 0, src0Channels));
      copySrc0Dispatch.setName(name());
      copySrc0Dispatch.setConfig(fmt::format(
          "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}",
          config0.invocC, config0.invocW, config0.invocH, config0.wgC,
          config0.wgW, config0.wgH));
      copySrc0Dispatch.setSourcePath(m_srcPath);
    }

    {
      const auto &src1 = opGraph.get(src1Id);

      uint32_t src1Channels = static_cast<uint32_t>(src1.channels.constant());

      uint8_t config1Key = (configEnc >> 16) & 0xFF;
      const Config &config1 = m_configs[config1Key];
      auto shader1 = copy_transform_compile(
          m_compiler, m_srcPath, 0, src1Channels, src0Channels, dstChannels,
          src1.format, dst.format, src0.channels.constant() % 8 == 0, config1);

      std::uint32_t tileX = config1.invocC * config1.wgC;
      std::uint32_t tileY = config1.invocW * config1.wgW;
      std::uint32_t tileZ = config1.invocH * config1.wgH;

      Sym workgroupCountX = symGraph.cdiv(src1.channels, tileX, false, false);
      Sym workgroupCountY = symGraph.cdiv(src1.width, tileY, false, false);
      Sym workgroupCountZ = symGraph.cdiv(src1.height, tileZ, false, false);

      auto copySrc1Dispatch =
          impl.registerDispatch(std::move(shader1), workgroupCountX,
                                workgroupCountY, workgroupCountZ);
      copySrc1Dispatch.addBinding("SRC_SET", "SRC_BINDING", Access::ReadOnly,
                                  src1Id);
      copySrc1Dispatch.addBinding("DST_SET", "DST_BINDING", Access::WriteOnly,
                                  dstId);
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.width));
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.height));

      Sym reads = symGraph.mul(src1.width, src1.height,
                               src1Channels * size_of(src1.type));
      Sym writes = symGraph.mul(src1.width, src1.height,
                                src1Channels * size_of(dst.type));
      copySrc1Dispatch.setMemoryReads(reads);
      copySrc1Dispatch.setMemoryWrites(writes);
      copySrc1Dispatch.setFlops(Sym::Const(0));
      // copySrc1Dispatch.usesCoopmat(false);

      // r[64:67]=x
      copySrc1Dispatch.setOperation(fmt::format("r[{}:{},:,:]=x", src0Channels,
                                                src0Channels + src1Channels));
      copySrc1Dispatch.setConfig(fmt::format(
          "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}",
          config1.invocC, config1.invocW, config1.invocH, config1.wgC,
          config1.wgW, config1.wgH));
      copySrc1Dispatch.setName(name());
      copySrc1Dispatch.setSourcePath(m_srcPath);
    }
  }
}

memory::string CopyTransformShader::name() const { return "copy-transform"; }

} // namespace denox::compiler::shaders
