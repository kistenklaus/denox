#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/vector.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class MemorySliceShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  MemorySliceShader() {
    {
      Pattern p;
      auto in = p.matchNode();
      auto slice = in->matchOutgoing();
      auto out = slice->matchDst();
      slice->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Slice; });

      m_patternHandles.emplace_back(in, std::move(slice), out);
      m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                           std::move(out));
    }
  }

  memory::optional<unsigned int> acceptMatch(
      [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
          &opGraph,
      unsigned int pattern,
      [[maybe_unused]] const algorithm::ConstGraphMatch<
          TensorInstance, ComputeOp> &match) const final override {
    const auto &patternHandle = m_patternHandles[pattern];
    memory::NodeId inId = match[patternHandle.in];
    memory::NodeId outId = match[patternHandle.out];
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

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int pattern,
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match) const final override {
    const auto &patternHandle = m_patternHandles[pattern];
    memory::NodeId inId = match[patternHandle.in];
    memory::NodeId outId = match[patternHandle.out];
    const auto &in = opGraph.get(inId);
    const auto &out = opGraph.get(outId);

    auto dispatch = impl.dispatch({});
    dispatch.addBinding(inId);
    dispatch.addBinding(outId);
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
    dispatch.addPushConstant(PushConstant::Dynamic(out.extent.x));
    dispatch.addPushConstant(PushConstant::Dynamic(out.extent.y));
    dispatch.setName(name(pattern));
    dispatch.setSourcePath(m_srcPath);
  }

  memory::string
  name([[maybe_unused]] unsigned int pattern) const final override {
    return "memory-slice";
  }

private:
  ShaderCapabilities m_capabilities;

  struct Handles {
    Pattern::NP in;
    Pattern::EP slice;
    Pattern::NP out;
  };
  memory::vector<Handles> m_patternHandles;

  io::Path m_srcPath = io::Path::cwd() / "compiler/shaders/slice/memory_slice.comp";
};

} // namespace denox::compiler::shaders
