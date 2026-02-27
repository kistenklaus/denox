#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <cassert>
#include <limits>

namespace denox::compiler::shaders {

struct DirectConvConfigCM {
  unsigned int cm_m;
  unsigned int cm_k;
  unsigned int cm_n;
  unsigned int wg_m;
  unsigned int wg_n;
  unsigned int sg_m;
  unsigned int sg_k;
  unsigned int sg_n;
  bool async;

  uint32_t subgroupSize;
};

class DirectConvShaderCM final : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

public:
  DirectConvShaderCM(spirv::GlslCompiler *compiler,
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
    memory::optional<Pattern::EP> upsample;
    Pattern::EP conv;
    memory::optional<Pattern::EP> relu;
    memory::optional<Pattern::EP> maxpool;
    Pattern::NP conv_out;
    Pattern::NP out;
  };

private:
  spirv::GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::assets() /
      "compiler/src/denox/compiler/implement/shaders/conv/direct_conv_cm.comp";
  bool m_subgroupControl;

  std::vector<DirectConvConfigCM> m_configs;

  unsigned int m_conv_pattern = std::numeric_limits<unsigned int>::max();
  unsigned int m_conv_activation_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_upsample_conv_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_upsample_conv_activation_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_conv_maxpool_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_conv_activation_maxpool_pattern =
      std::numeric_limits<unsigned int>::max();
};

} // namespace denox::compiler::shaders
