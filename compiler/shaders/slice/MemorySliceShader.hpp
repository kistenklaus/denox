#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class MemorySliceShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;

  MemorySliceShader() {
    {
      Pattern p;
      auto in = p.matchNode();
      auto slice = in->matchOutgoing();
      auto out = slice->matchDst();
      slice->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Slice; });

      m_capabilities.patterns.emplace_back(std::move(p), std::move(in),
                                           std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement([[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<
                     ComputeTensor, ComputeOp> &match) const final override {}

  memory::string name() const final override {
    return "memory-slice";
  }

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
