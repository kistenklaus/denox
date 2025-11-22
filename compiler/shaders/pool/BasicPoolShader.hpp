#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/vector.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/IShader.hpp"

namespace denox::compiler::shaders {

class BasicPoolShader final : public IShader {
  static constexpr std::size_t HWC_HWC_F16_PATTERN = 0;
  static constexpr std::size_t CHWC8_CHWC8_F16_PATTERN = 1;

public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicPoolShader(GlslCompiler *compiler);

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 unsigned int pattern,
                 unsigned int config,
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match, SymGraph& symGraph) const final override;

  memory::string name(unsigned int pattern, unsigned int config) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP pool;
    Pattern::NP out;
  };

private:
  GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/pool/basic_pool.comp";
};

} // namespace denox::compiler::shaders
