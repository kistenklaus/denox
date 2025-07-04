#include "./Conv2d.hpp"

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "vkcnn/device/tensor/WeightTensor.hpp"
#include "vkcnn/host/conv2d/Conv2dSource.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cassert>
#include <stdexcept>

namespace vkcnn::device::conv2d {

static merian::PipelineHandle
createPipeline(const merian::ContextHandle &context,
               const merian::ShaderCompilerHandle &shaderCompiler,
               const host::conv2d::Conv2dSource &src) {
  merian::PipelineLayoutBuilder pipelineLayoutBuilder{context};

  // input / output storage buffer descriptor set.
  merian::DescriptorSetLayoutBuilder set0Builder;
  set0Builder.add_binding_storage_buffer(); // input tensor.
  set0Builder.add_binding_storage_buffer(); // output tensor.

  if (!src.inlinedWeights()) {
    // constant weight uniform buffer descriptor set.
    set0Builder.add_binding_storage_buffer(); // weight tensor (binding = 2)
  }

  pipelineLayoutBuilder.add_descriptor_set_layout(
      set0Builder.build_push_descriptor_layout(context));

  pipelineLayoutBuilder.add_push_constant<glm::uvec2>();

  merian::PipelineLayoutHandle pipelineLayout =
      pipelineLayoutBuilder.build_pipeline_layout();

  merian::ShaderModuleHandle computeShaderModule;
  switch (src.lang()) {
  case ShaderLang::GLSL: {
    std::string src_name{src.debugName()};
    std::string src_string{src.src().begin(), src.src().end()};
    computeShaderModule = shaderCompiler->compile_glsl_to_shadermodule(
        context, src_string, src_name, vk::ShaderStageFlagBits::eCompute);
    break;
  }
  default:
    throw std::runtime_error("Shaderlang is not fully supported yet.");
  }

  return std::make_shared<merian::ComputePipeline>(pipelineLayout,
                                                   computeShaderModule);
}

Conv2d::Conv2d(const merian::ContextHandle &context,
               const merian::ShaderCompilerHandle &shaderCompiler,
               const host::conv2d::Conv2dSource &src)
    : m_tileSize(src.tileSize()) {
  if (!src.inlinedWeights()) {
    m_weightTensor = WeightTensor(context, src.weightTensor());
  }
  m_pipeline = createPipeline(context, shaderCompiler, src);
}

Conv2d::Conv2d(const merian::ContextHandle &context,
               const merian::ShaderCompilerHandle &shaderCompiler,
               host::conv2d::Conv2dSource &&src)
    : m_tileSize(src.tileSize()) {
  if (!src.inlinedWeights()) {
    m_weightTensor = WeightTensor(context, std::move(src.weightTensor()));
  }
  m_pipeline = createPipeline(context, shaderCompiler, src);
}

void Conv2d::dispatch(const merian::CommandBufferHandle &cmd,
                      glm::uvec2 inputImageSize, const ImageTensor &inputTensor,
                      const ImageTensor &outputTensor) {
  if (m_weightTensor) {
    // very cheap except for the first time.
    // m_weightTensor.flush(cmd);
  }

  cmd->bind(m_pipeline);
  cmd->push_constant<glm::uvec2>(m_pipeline, inputImageSize);
  if (m_weightTensor) {
    cmd->push_descriptor_set(m_pipeline, 0, inputTensor.get(), outputTensor.get(), 
        m_weightTensor.get());
  } else {
    cmd->push_descriptor_set(m_pipeline, 0, inputTensor.get(),
                             outputTensor.get());
  }
  glm::uvec2 workgroupCount =
      (inputImageSize + m_tileSize - glm::uvec2(1)) / m_tileSize;
  cmd->dispatch(workgroupCount.x, workgroupCount.y);
}

} // namespace vkcnn::device::conv2d
