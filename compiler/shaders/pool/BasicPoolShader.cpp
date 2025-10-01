#include "shaders/pool/BasicPoolShader.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/PoolFunction.hpp"
#include <stdexcept>

namespace denox::compiler::shaders {

BasicPoolShader::BasicPoolShader(GlslCompiler *compiler)
    : m_compiler(compiler) {
  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    return tensor.layout == memory::ActivationLayout::HWC ||
           tensor.layout == memory::ActivationLayout::CHWC8;
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
      if (op.tag() != ComputeOpTag::Pool) {
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
memory::optional<unsigned int> BasicPoolShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  const auto &in = opGraph.get(match[patternHandles.in]);
  const auto &out = opGraph.get(match[patternHandles.out]);
  memory::ActivationLayout inLayout = in.layout;
  memory::ActivationLayout outLayout = out.layout;

  if (memory::ActivationLayout::demote(outLayout, out.channels) !=
      memory::ActivationLayout::demote(inLayout, in.channels)) {
    return memory::nullopt;
  }
  if (in.type != memory::Dtype::F16) {
    return memory::nullopt;
  }
  if (out.type != memory::Dtype::F16) {
    return memory::nullopt;
  }
  return pattern;
}
void BasicPoolShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int pattern,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  const auto &patternHandles = m_patternHandles[pattern];
  memory::NodeId inId = match[patternHandles.in];
  memory::NodeId outId = match[patternHandles.out];
  memory::EdgeId poolId = match[patternHandles.pool];
  const auto &in = opGraph.get(inId);
  const auto &out = opGraph.get(outId);
  const auto &pool = opGraph.get(poolId);

  auto shader = m_compiler->read(m_srcPath);

  memory::ActivationLayout inLayout = in.layout;
  memory::ActivationLayout outLayout = out.layout;

  if (inLayout == memory::ActivationLayout::HWC &&
      outLayout == memory::ActivationLayout::HWC &&
      (in.channels % 8 != 0 || out.channels % 8 != 0)) {
    shader.define("istype", "uint16_t");
    shader.define("ISTYPE_SIZE", 2);
    shader.define("ostype", "uint16_t");
    shader.define("OSTYPE_SIZE", 2);
    shader.define("IN_LAYOUT_HWC");
    shader.define("OUT_LAYOUT_HWC");

    // decent defaults:
    if (in.channels <= 24) {
      unsigned int ix = (in.channels + 4 - 1) / 4;
      shader.define("INVOC_C", ix);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 4);
      shader.define("WG_W", 64);
      shader.define("WG_H", 1);
    } else if (in.channels <= 48) {
      unsigned int ix = (in.channels + 8 - 1) / 8;
      shader.define("INVOC_C", ix);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 8);
      shader.define("WG_W", 32);
      shader.define("WG_H", 1);
    } else {
      shader.define("INVOC_C", 2);
      shader.define("INVOC_W", 2);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 16);
      shader.define("WG_W", 16);
      shader.define("WG_H", 1);
    }
  } else if (inLayout == memory::ActivationLayout::HWC &&
             outLayout == memory::ActivationLayout::HWC &&
             (in.channels % 8 == 0 && out.channels % 8 == 0)) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_HWC8");
    shader.define("OUT_LAYOUT_HWC8");

    if (in.channels >= 32) {
      shader.define("INVOC_C", 8);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 4);
      shader.define("WG_W", 64);
      shader.define("WG_H", 1);
    } else if (in.channels >= 16) {
      shader.define("INVOC_C", 8);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 2);
      shader.define("WG_W", 128);
      shader.define("WG_H", 1);
    } else {
      shader.define("INVOC_C", 8);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 1);
      shader.define("WG_W", 256);
      shader.define("WG_H", 1);
    }
  } else if (inLayout == memory::ActivationLayout::CHWC8 &&
             outLayout == memory::ActivationLayout::CHWC8) {
    shader.define("istype", "uvec4");
    shader.define("ISTYPE_SIZE", 16);
    shader.define("ostype", "uvec4");
    shader.define("OSTYPE_SIZE", 16);
    shader.define("IN_LAYOUT_CHWC8");
    shader.define("OUT_LAYOUT_CHWC8");
    {
      shader.define("INVOC_C", 8);
      shader.define("INVOC_W", 1);
      shader.define("INVOC_H", 1);
      shader.define("WG_C", 1);
      shader.define("WG_W", 256);
      shader.define("WG_H", 1);
    }
  } else {
    throw std::logic_error("Invalid state");
  }

  assert(in.channels == out.channels);
  shader.define("CH", in.channels);

  shader.define("KERNEL_X", pool.pool()->kernelSize.x);
  shader.define("KERNEL_Y", pool.pool()->kernelSize.y);
  shader.define("STRIDE_X", pool.pool()->stride.x);
  shader.define("STRIDE_Y", pool.pool()->stride.y);
  shader.define("PADDING_X", pool.pool()->padding.x);
  shader.define("PADDING_Y", pool.pool()->padding.y);

  auto dispatch = impl.registerDispatch(std::move(shader));
  dispatch.addBinding(0, 0, AccessFlag::ReadOnly, inId);
  dispatch.addBinding(0, 1, AccessFlag::WriteOnly, outId);
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
  dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
  dispatch.setName(name(pattern));
  dispatch.setSourcePath(m_srcPath);
}
memory::string BasicPoolShader::name(unsigned int) const {
  return "basic-pool";
}
} // namespace denox::compiler::shaders
