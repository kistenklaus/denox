#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "vkcnn/common/shader/UpsampleShaderSource.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include <glm/fwd.hpp>

namespace vkcnn::runtime {

class UpsamplePipeline {
public:
  UpsamplePipeline(const ::merian::ContextHandle &context,
                   const ::merian::ShaderCompilerHandle &shaderCompiler,
                   const UpsampleShaderSource &source);

  void run(const ::merian::CommandBufferHandle &cmd,
           const ActivationDeviceTensor &input,
           const ActivationDeviceTensor &output);

private:
  glm::uvec3 m_tileSize;
  ::merian::PipelineHandle m_pipe;
};

} // namespace vkcnn::runtime
