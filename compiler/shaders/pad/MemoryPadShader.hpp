#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/optional.hpp"
#include "shaders/GlslCompiler.hpp"
#include "shaders/IShader.hpp"

namespace denox::compiler::shaders {

class MemoryPadShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  MemoryPadShader(GlslCompiler *compiler);

  memory::optional<unsigned int>
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
                 const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                     &match) const final override;

  memory::string name(unsigned int pattern) const final override;

private:

  GlslCompiler* m_compiler;

  ShaderCapabilities m_capabilities;
  struct Handles {
    Pattern::NP in;
    Pattern::EP pad;
    Pattern::NP out;
  };
  memory::vector<Handles> m_patternHandles;

  io::Path m_srcPath = io::Path::cwd() / "compiler/shaders/pad/memory_pad.comp";
};

} // namespace denox::compiler::shaders
