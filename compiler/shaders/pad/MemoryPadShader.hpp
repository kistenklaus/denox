#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/optional.hpp"
#include "shaders/IShader.hpp"
#include <fmt/base.h>
namespace denox::compiler::shaders {

class MemoryPadShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  MemoryPadShader() {
    {
      Pattern p;
      auto in = p.matchNode();
      auto pad = in->matchOutgoing();
      auto out = pad->matchDst();

      pad->matchRank(1);
      pad->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Pad; });

      m_patternHandles.emplace_back(in, std::move(pad), out);
      m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                           std::move(out));
    }
  }
  memory::optional<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override {
    const auto &patternHandles = m_patternHandles[pattern];

    memory::NodeId inId = match[patternHandles.in];
    memory::NodeId outId = match[patternHandles.out];
    const auto &in = opGraph.get(inId);
    const auto &out = opGraph.get(outId);
    if (in.layout != out.layout) {
      return memory::nullopt;
    }
    return pattern;
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int pattern,
                 const algorithm::ConstGraphMatch<
                     TensorInstance, ComputeOp> &match) const final override {
    const auto &patternHandles = m_patternHandles[pattern];
    memory::NodeId inId = match[patternHandles.in];
    memory::NodeId outId = match[patternHandles.out];
    
    auto dispatch = impl.dispatch({});
    dispatch.addBinding(inId);
    dispatch.addBinding(outId);

    const auto &in = opGraph.get(inId);
    const auto &out = opGraph.get(outId);
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
    dispatch.addPushConstant(PushConstant::Dynamic(out.extent.x));
    dispatch.addPushConstant(PushConstant::Dynamic(out.extent.y));
    dispatch.setName(name(pattern));
    dispatch.setSourcePath(m_srcPath);
  }

  memory::string
  name([[maybe_unused]] unsigned int pattern) const final override {
    return "memory-pad";
  }

private:
  ShaderCapabilities m_capabilities;
  struct Handles {
    Pattern::NP in;
    Pattern::EP pad;
    Pattern::NP out;
  };
  memory::vector<Handles> m_patternHandles;

  io::Path m_srcPath = io::Path::cwd() / "compiler/shaders/pad/memory_pad.comp";
};

} // namespace denox::compiler::shaders
