#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <cassert>
#include <limits>

namespace denox::compiler::shaders {

struct ConvConvAddConfig {
  unsigned int cm_m;
  unsigned int a_cm_k;
  unsigned int b_cm_k;
  unsigned int cm_n;
  unsigned int wg_m;
  unsigned int wg_n;
  unsigned int sg_m;
  unsigned int a_sg_k;
  unsigned int b_sg_k;
  unsigned int sg_n;
  bool a_async;
  bool b_async;
  uint32_t subgroupSize;
};

class ConvConvAddCMShader final : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

public:
  ConvConvAddCMShader(spirv::GlslCompiler *compiler,
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
    Pattern::NP a;
    Pattern::EP convA;
    Pattern::NP lhs;
    Pattern::NP b;
    Pattern::EP convB;
    Pattern::NP rhs;
    Pattern::EP add;
    Pattern::NP out;
    memory::optional<Pattern::EP> upsample;
    memory::optional<Pattern::EP> relu;
  };

private:
  spirv::GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::assets() /
      "compiler/src/denox/compiler/implement/shaders/conv/conv_conv_add_cm.comp";
  bool m_subgroupControl;
  unsigned int m_optimizationLevel;
  memory::vector<ConvConvAddConfig> m_configs;

  unsigned int m_conv_conv_add_patternn =
      std::numeric_limits<unsigned int>::max();
  unsigned int m_conv_conv_add_activation_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_A_upsample_conv_conv_add_pattern =
      std::numeric_limits<unsigned int>::max();
  unsigned int m_B_upsample_conv_conv_add_pattern =
      std::numeric_limits<unsigned int>::max();

  unsigned int m_A_upsample_conv_conv_add_activation_pattern =
      std::numeric_limits<unsigned int>::max();
  unsigned int m_B_upsample_conv_conv_add_activation_pattern =
      std::numeric_limits<unsigned int>::max();
};

} // namespace denox::compiler::shaders
