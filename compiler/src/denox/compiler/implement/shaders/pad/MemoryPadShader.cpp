#include "denox/compiler/implement/shaders/pad/MemoryPadShader.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include <iostream>

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

  auto fd = io::File::open(io::Path::assets() /
                               "compiler/src/denox/compiler/implement/shaders/"
                               "pad/memory_pad.configs",
                           io::File::OpenMode::Read);
  std::string str;
  str.resize(fd.size());
  fd.read_exact(std::span<std::byte>(reinterpret_cast<std::byte *>(str.data()),
                                     str.size()));
  std::stringstream ss(str);

  Config config;
  while (ss >> config.invocC >> config.invocW >> config.invocH >> config.wgC >>
         config.wgW >> config.wgH) {

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
      std::cerr << "Warning: MemoryPadShader implements non vectorized layouts "
                   "for format, "
                   "which may be vectorized, this works, but is suboptimal!"
                << std::endl;
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

  Config config = m_configs[configKey];
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
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", config.invocC,
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
