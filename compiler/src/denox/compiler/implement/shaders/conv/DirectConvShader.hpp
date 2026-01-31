#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <cassert>

namespace denox::compiler::shaders {

class DirectConvShader final : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr unsigned int CONV_PATTERN = 0;
  static constexpr unsigned int CONV_ACTIVATION_PATTERN = 1;

public:
  DirectConvShader(spirv::GlslCompiler *compiler,
                   const CompileOptions &options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  std::size_t parameterMemorySize(
      const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
      unsigned int pattern,
      const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match)
      const final override;

  void
  implement(OpImpl &impl,
            const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
            unsigned int pattern, unsigned int config,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
            SymGraph &symGraph) const final override;

  memory::string name() const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP conv;
    memory::optional<Pattern::EP> relu;
    Pattern::NP out;
  };

private:
  spirv::GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::assets() /
      "compiler/src/denox/compiler/implement/shaders/conv/direct_conv.comp";
  bool m_enableConvReluFusion;

  unsigned int m_subgroupSize;
  uint32_t m_maxComputeWorkGroupInvocations;
  std::array<uint32_t, 3> m_maxComputeWorkGroupSize;
};

} // namespace denox::compiler::shaders
