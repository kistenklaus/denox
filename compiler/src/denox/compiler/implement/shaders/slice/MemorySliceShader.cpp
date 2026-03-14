#include "denox/compiler/implement/shaders/slice/MemorySliceShader.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/io/fs/File.hpp"
#include <iostream>
#include <stdexcept>

namespace denox::compiler::shaders {

MemorySliceShader::MemorySliceShader(spirv::GlslCompiler *compiler,
                                     const CompileOptions &options)
    : m_compiler(compiler),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize),
      m_maxComputeWorkGroupInvocations(
          options.deviceInfo.limits.maxComputeWorkGroupInvocations),
      m_maxComputeWorkGroupSize(
          options.deviceInfo.limits.maxComputeWorkGroupSize),
      m_optimizationLevel(options.optimizationLevel) {

  auto fd = io::File::open(io::Path::assets() /
                               "compiler/src/denox/compiler/implement/shaders/"
                               "slice/memory_slice.configs",
                           io::File::OpenMode::Read);
  std::string str;
  str.resize(fd.size());
  fd.read_exact(std::span<std::byte>(reinterpret_cast<std::byte *>(str.data()),
                                     str.size()));
  std::stringstream ss(str);

  while (!ss.eof()) {
    Config config;
    ss >> config.invocC;
    ss >> config.invocW;
    ss >> config.invocH;
    ss >> config.wgC;
    ss >> config.wgW;
    ss >> config.wgH;

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

  if (in.type != TensorDataType::Float16) {
    return {};
  }
  if (out.type != TensorDataType::Float16) {
    return {};
  }

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
  for (unsigned int c = 0; c < m_configs.size(); ++c) {
    const auto &config = m_configs[c];
    switch (variant) {
    case HWC:

      if (m_optimizationLevel < 3) {
        if (config.invocW != 1) {
          continue;
        }
        if (config.wgH != 1) {
          continue;
        }
      }

      promissing.push_back(c);
      break;
    case HWC8:
      if (config.invocC % 8 != 0) {
        continue;
      }
      if (m_optimizationLevel < 3) {
        if (config.invocC != 8) {
          continue;
        }
        if (config.wgH != 1) {
          continue;
        }
      }
      promissing.push_back(c);
      break;
    case CHWC8:
      if (config.invocC % 8 != 0) {
        continue;
      }
      if (m_optimizationLevel < 3) {
        if (config.invocC != 8) {
          continue;
        }
        if (config.wgH != 1) {
          continue;
        }
        if (config.wgC != 1) {
          continue;
        }
      }
      promissing.push_back(c);
      break;
    }
  }
  return promissing;
}

static spirv::GlslCompilerInstance
memory_slice_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                     TensorFormat inputFormat, TensorFormat outputFormat,
                     unsigned int channels,
                     const MemorySliceShader::Config &config) {
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
      std::cerr << "Warning: MemorySliceShader implements non vectorized "
                   "layouts for format, "
                   "which may be vectorized, this works, but is suboptimal!"
                << std::endl;
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
  shader.define("WG_C", config.wgC);
  shader.define("WG_W", config.wgW);
  shader.define("WG_H", config.wgH);
  return shader;
}

void MemorySliceShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {

  const auto &patternHandle = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandle.in];
  memory::NodeId outId = match[patternHandle.out];
  memory::EdgeId opId = match[patternHandle.slice];

  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &slice = opGraph.get(opId).slice();

  assert(in.channels == out.channels);

  uint32_t C = static_cast<uint32_t>(in.channels.constant());

  enum Variant {
    HWC,
    HWC8,
    CHWC8,
  };

  Config config = m_configs[configKey];

  auto shader = memory_slice_compile(m_compiler, m_srcPath, in.format,
                                     out.format, C, config);

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

  dispatch.addPushConstant(PushConstant::Dynamic(slice->top));
  dispatch.addPushConstant(PushConstant::Dynamic(slice->left));
  dispatch.addPushConstant(PushConstant::Dynamic(in.width));
  dispatch.addPushConstant(PushConstant::Dynamic(in.height));
  dispatch.addPushConstant(PushConstant::Dynamic(out.width));
  dispatch.addPushConstant(PushConstant::Dynamic(out.height));
  dispatch.setName(name());
  dispatch.setOperation(fmt::format(
      "x[:,{}:{},{}:{}]",
      slice->top.isConstant() ? fmt::format("{}", slice->top.constant())
                              : "<dyn>",
      in.height.isConstant() ? fmt::format("{}", in.height.constant())
                             : "<dyn>",
      slice->left.isConstant() ? fmt::format("{}", slice->left.constant())
                               : "<dyn>",
      in.width.isConstant() ? fmt::format("{}", in.width.constant())
                            : "<dyn>"));
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", config.invocC,
      config.invocW, config.invocW, config.wgC, config.wgW, config.wgH));
  dispatch.setSourcePath(m_srcPath);

  Sym reads = symGraph.mul(in.width, in.height, C * size_of(in.type));
  Sym writes = symGraph.mul(out.width, out.height, C * size_of(out.type));
  dispatch.setMemoryReads(reads);
  dispatch.setMemoryWrites(writes);
  dispatch.setFlops(Sym::Const(0));
  dispatch.usesCoopmat(false);
}

memory::string MemorySliceShader::name() const { return "memory-slice"; }
} // namespace denox::compiler::shaders
