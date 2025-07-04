#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "vkcnn/ImageTensor.hpp"
#include "vkcnn/host/ShaderSource.hpp"
#include <cassert>

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include <glm/fwd.hpp>
#include <glm/vec2.hpp>

#include <vector>
namespace vkcnn::device {

class OpComputePipe {
private:
public:
  OpComputePipe(const merian::ContextHandle &context,
                const merian::ShaderCompilerHandle &shaderCompiler,
                const ShaderSource &src,
                std::span<const merian::BufferHandle> constants,
                glm::uvec2 tileSize,
                std::string_view debugName = "OpCompute");

  void run(const merian::CommandBufferHandle &cmd,
           const ::vkcnn::ImageTensor &input,
           const ::vkcnn::ImageTensor &output);

private:
  merian::PipelineHandle static createPipeline(
      const merian::ContextHandle &context,
      const merian::ShaderCompilerHandle &shaderCompiler,
      const ShaderSource &src, unsigned int constantBuffers,
      std::string_view debugName);

  merian::PipelineHandle m_pipeline;
  std::vector<merian::BufferHandle> m_constantBuffers;
  std::vector<vk::DescriptorBufferInfo> m_bufferInfos;
  std::vector<vk::WriteDescriptorSet> m_descriptorSets;
  glm::uvec2 m_tileSize;
};

} // namespace vkcnn::device
