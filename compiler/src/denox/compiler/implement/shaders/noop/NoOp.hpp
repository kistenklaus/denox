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
  implement([[maybe_unused]] OpImpl &impl,
            [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
                &opGraph,
            [[maybe_unused]] unsigned int pattern,
            [[maybe_unused]] unsigned int config,
            [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance,
                                                              ComputeOp> &match,
            [[maybe_unused]] SymGraph &symGraph) const final override {}

  memory::string name() const final override { return "noop(x)"; }

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
