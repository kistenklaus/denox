#pragma once

#include "Options.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/IShader.hpp"
#include <cassert>

namespace denox::compiler::shaders {

class DirectConvShader final : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr unsigned int CONV_PATTERN = 0;
  static constexpr unsigned int CONV_ACTIVATION_PATTERN = 1;

public:
  DirectConvShader(GlslCompiler *compiler, const Options& options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  std::size_t parameterMemorySize(
      const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
      unsigned int pattern,
      const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match)
      const final override;

  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int pattern,
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match) const final override;

  memory::string name(unsigned int pattern) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP conv;
    memory::optional<Pattern::EP> relu;
    Pattern::NP out;
  };

private:
  GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/conv/direct_conv.comp";
  bool m_enableConvReluFusion;
  unsigned int m_subgroupSize;
};

} // namespace denox::compiler::shaders
