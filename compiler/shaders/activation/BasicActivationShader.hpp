#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "compiler/ir/TensorInstance.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/IShader.hpp"

namespace denox::compiler::shaders {

class BasicActivationShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr std::size_t ACTI_FUNC_TYPE_MASK = 0xFF << 8;
  static constexpr std::size_t ACTI_FUNC_TYPE_ReLU = 1 << 8;

  BasicActivationShader(GlslCompiler *compiler, const Options& options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::optional<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int patternEnc,
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match, SymGraph& symGraph) const final override;

  memory::string name(unsigned int pattern) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP acti;
    Pattern::NP out;
  };

private:
  GlslCompiler *m_compiler;

  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/activation/basic_activation.comp";

  unsigned m_subgroupSize;
};

} // namespace denox::compiler::shaders
