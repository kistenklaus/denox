#include "denox/compiler/implement/shaders/add/BasicAddShader.hpp"
#include "denox/common/ActivationFunction.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"

namespace denox::compiler::shaders {

BasicAddShader::BasicAddShader(spirv::GlslCompiler *compiler,
                               const CompileOptions &options)
    : m_compiler(compiler), m_optimizationLevel(options.optimizationLevel) {

  auto fd = io::File::open(
      io::Path::assets() /
          "compiler/src/denox/compiler/implement/shaders/add/basic_add.configs",
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
    if (tensor.format != TensorFormat::SSBO_HWC &&
        tensor.format != TensorFormat::SSBO_CHWC8) {
      return false;
    }
    return true;
  };
  const auto isAdd = [](const ComputeOp &op) {
    return op.tag() == ComputeOpKind::Add;
  };
  {
    Pattern addPattern;
    auto add = addPattern.matchEdge();
    add->matchRank(2);
    add->matchValue(isAdd);
    auto in0 = add->matchSrc(0);
    auto in1 = add->matchSrc(1);
    auto out = add->matchDst();
    in0->matchValue(supportedTensor);
    in1->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    m_patternHandles.emplace_back(in0, in1, memory::nullopt, out);
    m_capabilities.patterns.emplace_back(std::move(addPattern), std::move(in0),
                                         std::move(in1), std::move(out));
  }
  if (options.features.enableConvReluFusion) {
    Pattern addReluPattern;
    auto add = addReluPattern.matchEdge();
    add->matchRank(2);
    add->matchValue(isAdd);
    auto in0 = add->matchSrc(0);
    auto in1 = add->matchSrc(1);
    auto x = add->matchDst();
    auto acti = x->matchOutgoing();
    auto out = acti->matchDst();
    acti->matchRank(1);
    acti->matchValue([](const ComputeOp &op) {
      if (op.tag() != ComputeOpKind::Activation) {
        return false;
      }
      const auto &func = op.activation().func;
      return (func.kind() == ActivationFunctionKind::ReLU) ||
             (func.kind() == ActivationFunctionKind::LeakyReLU);
    });
    in0->matchValue(supportedTensor);
    in1->matchValue(supportedTensor);
    out->matchValue(supportedTensor);
    m_patternHandles.emplace_back(in0, in1, acti, out);
    m_capabilities.patterns.emplace_back(std::move(addReluPattern),
                                         std::move(in0), std::move(in1),
                                         std::move(out));
  }
}

memory::vector<unsigned int> BasicAddShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in0 = opGraph.get(match[patternHandles.in0]);
  const auto &in1 = opGraph.get(match[patternHandles.in1]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (in0.format != out.format || in1.format != out.format) {
    return {};
  }
  if (in0.channels != out.channels || in1.channels != out.channels) {
    diag::invalid_state();
  }

  const bool vec = out.channels.constant() % 8 == 0 ||
                   out.format == TensorFormat::SSBO_CHWC8;
  memory::vector<unsigned int> promissing;
  for (unsigned int c = 0; c < m_configs.size(); ++c) {
    const auto &config = m_configs[c];
    if (vec && config.invocC % 8 != 0) {
      continue;
    }
    if (m_optimizationLevel < 3) {
      if (config.wgH != 1) {
        continue;
      }
      if (vec && config.invocC != 8) {
        continue;
      }
      if (!vec && config.invocW != 1) {
        continue;
      }
      if (out.format == TensorFormat::SSBO_CHWC8 && config.wgC != 1) {
        continue;
      }
    }
    promissing.push_back(c);
  }
  return promissing;
}

static spirv::GlslCompilerInstance
basic_add_compile(spirv::GlslCompiler *compiler, const io::Path &srcPath,
                  TensorFormat format, unsigned int channels,
                  memory::optional<ActivationFunction> activationFunction,
                  const BasicAddShader::Config &config) {
  auto shader = compiler->read(srcPath);
  shader.define("CH", channels);

  if (activationFunction) {
    switch (activationFunction->kind()) {
    case ActivationFunctionKind::ReLU:
      shader.define("ACTIVATION_ReLU");
      break;
    case ActivationFunctionKind::LeakyReLU:
      shader.define("ACTIVATION_LeakyReLU");
      shader.define(
          "ACTIVATION_LeakyReLU_alpha",
          fmt::format("({}f)", activationFunction->leaky_relu().alpha));
      break;
    case ActivationFunctionKind::SiLU:
    case ActivationFunctionKind::Swish:
      diag::invalid_state();
    }
  } else {
    shader.define("ACTIVATION_NONE");
  }

  if (format == TensorFormat::SSBO_HWC && channels % 8 == 0 &&
      config.invocC % 8 == 0) {
    shader.define("istype", "uvec4");
    shader.define("ostype", "uvec4");
    shader.define("LAYOUT_HWC8");
  } else if (format == TensorFormat::SSBO_HWC) {
    shader.define("istype", "uint16_t");
    shader.define("ostype", "uint16_t");
    shader.define("LAYOUT_HWC");
  } else if (format == TensorFormat::SSBO_CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ostype", "uvec4");
    shader.define("LAYOUT_CHWC8");
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

void BasicAddShader::implement(
    OpImpl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern, unsigned int configKey,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId in0Id = match[patternHandles.in0];
  memory::NodeId in1Id = match[patternHandles.in1];
  memory::NodeId outId = match[patternHandles.out];
  const auto &out = opGraph.get(outId);

  memory::optional<ActivationFunction> activationFunction;
  if (patternHandles.acti.has_value()) {
    activationFunction =
        opGraph.get(match[*patternHandles.acti]).activation().func;
  }

  const uint32_t C = static_cast<uint32_t>(out.channels.constant());
  const Config &config = m_configs[configKey];

  auto shader = basic_add_compile(m_compiler, m_srcPath, out.format, C,
                                  activationFunction, config);

  std::uint32_t tileC = config.invocC * config.wgC;
  std::uint32_t tileW = config.invocW * config.wgW;
  std::uint32_t tileH = config.invocH * config.wgH;

  Sym workgroupCountX = symGraph.cdiv(out.channels, tileC, false, false);
  Sym workgroupCountY = symGraph.cdiv(out.width, tileW, false, false);
  Sym workgroupCountZ = symGraph.cdiv(out.height, tileH, false, false);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding("INPUT0_SET", "INPUT0_BINDING", Access::ReadOnly, in0Id);
  dispatch.addBinding("INPUT1_SET", "INPUT1_BINDING", Access::ReadOnly, in1Id);
  dispatch.addBinding("OUTPUT_SET", "OUTPUT_BINDING", Access::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(out.width));
  dispatch.addPushConstant(PushConstant::Dynamic(out.height));
  dispatch.setSourcePath(m_srcPath);

  Sym bytes =
      symGraph.mul(symGraph.mul(out.width, out.height), C * size_of(out.type));
  dispatch.setMemoryReads(symGraph.mul(bytes, 2));
  dispatch.setMemoryWrites(bytes);
  dispatch.setFlops(symGraph.mul(symGraph.mul(out.width, out.height), C));

  dispatch.setName(name());
  dispatch.setConfig(fmt::format(
      "INVOC_C={}#INVOC_W={}#INVOC_H={}#WG_C={}#WG_W={}#WG_H={}", config.invocC,
      config.invocW, config.invocH, config.wgC, config.wgW, config.wgH));

  if (!activationFunction) {
    dispatch.setOperation("x+y");
  } else if (activationFunction->kind() == ActivationFunctionKind::ReLU) {
    dispatch.setOperation("relu(x+y)");
  } else {
    dispatch.setOperation(
        fmt::format("leaky_relu(x+y,alpha={})",
                    activationFunction->leaky_relu().alpha));
  }
}

memory::string BasicAddShader::name() const { return "basic-add"; }

} // namespace denox::compiler::shaders
