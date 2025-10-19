#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/vector.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/IShader.hpp"

namespace denox::compiler::shaders {

class MemorySliceShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  MemorySliceShader(GlslCompiler *compiler);

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
                     &match, SymGraph& symGraph) const final override;

  memory::string name(unsigned int pattern) const final override;

private:
  struct Handles {
    Pattern::NP in;
    Pattern::EP slice;
    Pattern::NP out;
  };

private:
  GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/slice/memory_slice.comp";
};

} // namespace denox::compiler::shaders
