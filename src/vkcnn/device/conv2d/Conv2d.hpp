#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "vkcnn/device/tensor/ImageTensor.hpp"
#include "vkcnn/device/tensor/WeightTensor.hpp"
#include "vkcnn/host/conv2d/Conv2dSource.hpp"
#include <glm/fwd.hpp>
#include <glm/vec2.hpp>

namespace vkcnn::device::conv2d {

class Conv2d {

public:
  Conv2d(const merian::ContextHandle &context,
         const merian::ShaderCompilerHandle &shaderCompiler,
         const host::conv2d::Conv2dSource &src);

  Conv2d(const merian::ContextHandle &context,
         const merian::ShaderCompilerHandle &shaderCompiler,
         host::conv2d::Conv2dSource &&src);

  void dispatch(const merian::CommandBufferHandle &cmd,
                glm::uvec2 inputImageSize,
                const ImageTensor &inputTensor,
                const ImageTensor &outputTensor);

private:
  glm::uvec2 m_tileSize;
  merian::PipelineHandle m_pipeline;
  WeightTensor m_weightTensor;
};

} // namespace vkcnn::device::conv2d
