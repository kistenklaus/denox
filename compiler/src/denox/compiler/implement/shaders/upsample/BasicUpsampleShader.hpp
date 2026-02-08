#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"
namespace denox::compiler::shaders {

class BasicUpsampleShader : public IShader {
private:
  struct Config_HWC {
    uint32_t invocC; // 1,2,3,4 (anything else doens't make any sense)
    // invocW = 1
    uint32_t invocH; // 1,2
    // wgC; // pick as (C + invocC - 1) /  invocC
    uint32_t wgSizeHint; // assume wgH = 1 and derive wgW from wgC (around upto
                         // multiples of 32)
  };

  struct Config_HWC8 {
    // invocC = 8
    // uint32_t invocW; // 1,2
    uint32_t invocW; // 1, 2
    uint32_t invocH; // 1,2
    // wgC; // pick as (C + 7) / 8
    uint32_t wgSizeHint; // assume wgH = 1 and derive from wgC
    // NOTE: prefer xdispatch == 1
  };

  struct Config_CHWC8 {
    // invocC = 8
    uint32_t invocW; // 1, 2
    uint32_t invocH; // 1, 2
    // wgC; <- i will claim that 1 is best!
    uint32_t wgSize; // => wgW == wgSize
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

  BasicUpsampleShader(spirv::GlslCompiler *compiler,
                      const CompileOptions &options);

  memory::vector<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override;

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

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
    Pattern::EP upsample;
    Pattern::NP out;
  };

private:
  spirv::GlslCompiler *m_compiler;
  ShaderCapabilities m_capabilities;
  memory::vector<Handles> m_patternHandles;
  io::Path m_srcPath = io::Path::assets() /
                       "compiler/src/denox/compiler/implement/shaders/"
                       "upsample/basic_upsample.comp";

  uint32_t m_subgroupSize;
  uint32_t m_maxComputeWorkGroupInvocations;
  // static std::array<BasicUpsampleConfig, 5> BASIC_UPSAMPLE_CONFIGS{
  std::array<uint32_t, 3> m_maxComputeWorkGroupSize;
  memory::vector<Config_HWC> m_hwc_configs;
  memory::vector<Config_HWC8> m_hwc8_configs;
  memory::vector<Config_CHWC8> m_chwc8_configs;

  uint32_t m_optimizationLevel;
};

} // namespace denox::compiler::shaders
