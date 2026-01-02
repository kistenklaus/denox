#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"

namespace denox::compiler {

class NoOp : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  NoOp();

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  void
  implement(OpImpl &impl,
            const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
            unsigned int pattern, unsigned int config,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
            SymGraph &symGraph) const final override {}

  memory::string name(unsigned int pattern,
                      unsigned int config) const final override {
    return "noop";
  }

private:
  struct Handles {
    Pattern::NP in;
    Pattern::NP out;
  };

private:
  ShaderCapabilities m_capabilities;
  Handles m_handles;


};

} // namespace denox::compiler
