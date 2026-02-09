#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/glsl/GlslCompiler.hpp"

namespace denox::compiler::shaders {

class BasicActivationShader : public IShader {
private:
  struct Config_HWC {
    uint32_t invocC;
    uint32_t invocH;
    uint32_t wgSizeHint;
  };
  struct Config_HWC8 {
    uint32_t invocW;
    uint32_t invocH;
    uint32_t wgSizeHint;
  };
  struct Config_CHWC8 {
    uint32_t invocW;
    uint32_t invocH;
    uint32_t wgSize;
  };

public:
  struct Config {
    uint32_t invocC;
    uint32_t invocW;
    uint32_t invocH;
    uint32_t wgC;
    uint32_t wgW;
    uint32_t wgH;
  };
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicActivationShader(spirv::GlslCompiler *compiler,
                        const CompileOptions &options);

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

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
    Pattern::EP acti;
    Pattern::NP out;
  };

private:
  spirv::GlslCompiler *m_compiler;

  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath = io::Path::assets() /
                       "compiler/src/denox/compiler/implement/shaders/"
                       "activation/basic_activation.comp";

  uint32_t m_subgroupSize;
  uint32_t m_maxComputeWorkGroupInvocations;
  std::array<std::uint32_t, 3> m_maxComputeWorkGroupSize;

  memory::vector<Config_HWC> m_hwc_configs;
  memory::vector<Config_HWC8> m_hwc8_configs;
  memory::vector<Config_CHWC8> m_chwc8_configs;

  uint32_t m_optimizationLevel;
};

} // namespace denox::compiler::shaders
