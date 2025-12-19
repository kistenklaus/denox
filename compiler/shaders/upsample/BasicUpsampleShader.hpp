#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/IShader.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
namespace denox::compiler::shaders {

class BasicUpsampleShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicUpsampleShader(GlslCompiler *compiler, const Options& options);

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  void
  implement(Impl &impl,
            const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
            unsigned int pattern, unsigned int config,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
            SymGraph &symGraph) const final override;

  memory::string name(unsigned int pattern, unsigned int config) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP upsample;
    Pattern::NP out;
  };

private:
  GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/upsample/basic_upsample.comp";

  uint32_t m_maxComputeWorkGroupInvocations;
  std::array<uint32_t, 3> m_maxComputeWorkGroupSize;
};

} // namespace denox::compiler::shaders
