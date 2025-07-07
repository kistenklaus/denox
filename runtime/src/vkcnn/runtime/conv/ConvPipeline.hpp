#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/FilterDeviceTensor.hpp"
#include <glm/fwd.hpp>

namespace vkcnn::runtime {

class ConvPipeline {
public:
  ConvPipeline(const ::merian::ContextHandle &context,
               const ::merian::ShaderCompilerHandle &shaderCompiler,
               const ConvShaderSource &source,
               const FilterDeviceTensor &filterWeights);

  void run(const ::merian::CommandBufferHandle &cmd,
           const ActivationDeviceTensor &input,
           const ActivationDeviceTensor &output);

private:
  glm::uvec2 m_tileSize;
  FilterDeviceTensor m_filterWeights;
  std::string name;

#ifndef NDEBUG
  ActivationLayout m_inputLayout;
  FloatType m_inputType;
  ActivationLayout m_outputLayout;
  FloatType m_outputType;
#endif
  ::merian::PipelineHandle m_pipe;
};

} // namespace vkcnn::runtime
