#include "shaders/activation/BasicActivationShader.hpp"
#include "diag/invalid_state.hpp"
#include "diag/unreachable.hpp"
#include "memory/dtype/dtype.hpp"
#include "model/ActivationFunction.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

BasicActivationShader::BasicActivationShader(GlslCompiler *compiler,
                                             const Options &options)
    : m_compiler(compiler),
      m_subgroupSize(options.deviceInfo.subgroup.subgroupSize) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::HWC &&
        tensor.layout != memory::ActivationLayout::CHWC8) {
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
      if (op.tag() != ComputeOpTag::Activation) {
        return false;
      }
      if (op.activation().func != ActivationFunction::ReLU) {
        return false;
      }
      return true;
    });
    m_patternHandles.emplace_back(in, std::move(acti), out);
    m_capabilities.patterns.emplace_back(std::move(basicPattern), std::move(in),
                                         std::move(out));
  }
}
memory::optional<unsigned int> BasicActivationShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  if (memory::ActivationLayout::demote(in.layout, in.channels) !=
      memory::ActivationLayout::demote(out.layout, out.channels)) {
    return memory::nullopt;
  }
  return pattern | ACTI_FUNC_TYPE_ReLU;
}
void BasicActivationShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int patternEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  unsigned int pattern = patternEnc & ~ACTI_FUNC_TYPE_MASK;
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &acti = opGraph.get(match[patternHandles.acti]).activation();

  auto shader = m_compiler->read(m_srcPath);
  shader.define("SG_SIZE", m_subgroupSize);
  assert(in.channels == out.channels);
  shader.define("CH", in.channels);
  shader.define("in_atype", "float16_t");
  shader.define("IN_ATYPE_SIZE", 2);
  shader.define("out_atype", "float16_t");
  shader.define("OUT_ATYPE_SIZE", 2);

  switch (acti.func) {
  case ActivationFunction::ReLU:
    shader.define("ACTIVATION_ReLU");
    break;
  case ActivationFunction::LeakyReLU:
  case ActivationFunction::SiLU:
    diag::invalid_state();
  }

  std::uint32_t wgC;
  std::uint32_t wgH;
  std::uint32_t wgW;
  std::uint32_t invocC;
  std::uint32_t invocW;
  std::uint32_t invocH;

  if (in.layout == memory::ActivationLayout::HWC &&
      out.layout == memory::ActivationLayout::HWC &&
      (in.channels % 8 != 0 || out.channels % 8 != 0)) {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);

    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");

    if (in.channels >= 16) {
      invocC = 2;
      invocW = 2;
      invocH = 1;
      wgC = 8;
      wgW = 32;
      wgH = 1;
    } else {
      invocC = 1;
      invocW = 4;
      invocH = 1;
      wgC = in.channels;
      wgW = 32;
      wgH = 1;
    }
  } else if (in.layout == memory::ActivationLayout::HWC &&
             out.layout == memory::ActivationLayout::HWC &&
             (in.channels % 8 == 0 && out.channels % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");

    if (in.channels >= 32) {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 4;
      wgW = 64;
      wgH = 1;
    } else if (in.channels >= 16) {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 2;
      wgW = 128;
      wgH = 1;
    } else {
      invocC = 8;
      invocW = 1;
      invocH = 1;
      wgC = 1;
      wgW = 256;
      wgH = 1;
    }
  } else if (in.layout == memory::ActivationLayout::CHWC8 &&
             out.layout == memory::ActivationLayout::CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);

    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");

    invocC = 8;
    invocW = 1;
    invocH = 1;
    wgC = 1;
    wgW = 256;
    wgH = 1;
  } else {
    compiler::diag::invalid_state();
  }
  shader.define("INVOC_C", invocC);
  shader.define("INVOC_W", invocW);
  shader.define("INVOC_H", invocH);
  shader.define("WG_C", wgC);
  shader.define("WG_W", wgW);
  shader.define("WG_H", wgH);

  std::uint32_t tileC = invocC * wgC;
  std::uint32_t tileW = invocW * wgW;
  std::uint32_t tileH = invocH * wgH;

  Sym workgroupCountX = symGraph.cdiv(in.channels, tileC);
  Sym workgroupCountY = symGraph.cdiv(in.extent.x.asSym(), tileW);
  Sym workgroupCountZ = symGraph.cdiv(in.extent.y.asSym(), tileH);

  auto dispatch = impl.registerDispatch(std::move(shader), workgroupCountX,
                                        workgroupCountY, workgroupCountZ);
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(patternEnc));
  dispatch.setSourcePath(m_srcPath);
}
memory::string BasicActivationShader::name(unsigned int pattern) const {
  switch (pattern & ACTI_FUNC_TYPE_MASK) {
  case ACTI_FUNC_TYPE_ReLU:
    return "relu";
  default:
    compiler::diag::unreachable();
  }
}
} // namespace denox::compiler::shaders
