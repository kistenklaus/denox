#include "denox/compiler/implement/shaders/activation/BasicActivationShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include <iostream>

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

  auto fd = io::File::open(io::Path::assets() /
                               "compiler/src/denox/compiler/implement/shaders/"
                               "activation/basic_activation.configs",
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
      std::cerr << "Warning: BasicActivationShader implements non vectorized "
                   "layouts for format, "
                   "which may be vectorized, this works, but is suboptimal!"
                << std::endl;
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

  Config config = m_configs[configKey];

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
