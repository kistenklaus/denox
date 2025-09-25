#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class CopyTransformShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;

  CopyTransformShader() { 
    {
      Pattern explicitConcat;
      auto concat = explicitConcat.matchEdge();
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement(unsigned int pattern,
                 const algorithm::ConstGraphMatch<ComputeTensor, ComputeOp>
                     &match) const final override {}

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
