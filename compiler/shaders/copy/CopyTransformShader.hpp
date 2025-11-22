#pragma once

#include "Options.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/IShader.hpp"

namespace denox::compiler::shaders {

class CopyTransformShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr unsigned int IMPLICIT_CONCAT_MODE = 1 << 8;
  static constexpr unsigned int SINGLE_COPY_CONCAT_MODE = 2 << 8;
  static constexpr unsigned int EXPLICIT_CONCAT_MODE = 3 << 8;
  static constexpr unsigned int CONCAT_MODE_MASK = 0xFF << 8;
  static constexpr unsigned int PATTERN_MASK = 0xFF;

  static constexpr bool
      ENABLE_UNSTABLE_FEATURE_IMPLICIT_CONCAT_LIFETIME_INFERANCE = false;

  CopyTransformShader(GlslCompiler *compiler, const Options& options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  float speedup(unsigned int patternEnc) const final override;

  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int pattern,
                 unsigned int config,
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match, SymGraph& symGraph) const final override;

  memory::string name(unsigned int pattern, unsigned int config) const final override;

private:
  struct Handles {
    Pattern::NP src0;
    Pattern::NP src1;
    Pattern::NP dst;
  };

private:
  GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/copy/copy_transform.comp";

  bool m_enableImplicitConcat;
};

} // namespace denox::compiler::shaders
