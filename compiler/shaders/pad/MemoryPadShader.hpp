#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/IShader.hpp"
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

      m_capabilities.patterns.emplace_back(std::move(p), std::move(in), std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement([[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<
                     TensorInstance, ComputeOp> &match) const final override {}

  memory::string name([[maybe_unused]] unsigned int pattern) const final override {
    return "memory-pad";
  }

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
