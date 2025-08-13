#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "vkcnn/common/shader/CopyShaderSource.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include <glm/fwd.hpp>

namespace vkcnn::runtime {

class CopyPipeline {
public:
  CopyPipeline(const ::merian::ContextHandle &context,
               const ::merian::ShaderCompilerHandle &shaderCompiler,
               const CopyShaderSource &source);

  void run(const ::merian::CommandBufferHandle &cmd,
           const ActivationDeviceTensor &input,
           const ActivationDeviceTensor &output);

private:
  glm::uvec3 m_tileSize;
  ::merian::PipelineHandle m_pipe;
};

} // namespace vkcnn::runtime
