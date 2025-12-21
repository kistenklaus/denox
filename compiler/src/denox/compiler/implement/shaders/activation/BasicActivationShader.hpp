#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"

namespace denox::compiler::shaders {

class BasicActivationShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicActivationShader(spirv::GlslCompiler *compiler, const Options &options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  void
  implement(Impl &impl,
            const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
            unsigned int pattern, unsigned int config,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
            SymGraph &symGraph) const final override;

  memory::string name(unsigned int pattern,
                      unsigned int config) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP acti;
    Pattern::NP out;
  };

private:
  spirv::GlslCompiler *m_compiler;

  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/activation/basic_activation.comp";

  uint32_t m_subgroupSize;
  uint32_t m_maxComputeWorkGroupInvocations;
  std::array<std::uint32_t, 3> m_maxComputeWorkGroupSize;
};

} // namespace denox::compiler::shaders
