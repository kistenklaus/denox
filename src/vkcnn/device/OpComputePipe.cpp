#include "./OpComputePipe.hpp"

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cassert>
#include <stdexcept>

merian::PipelineHandle vkcnn::device::OpComputePipe::createPipeline(
    const merian::ContextHandle &context,
    const merian::ShaderCompilerHandle &shaderCompiler, const ShaderSource &src,
    unsigned int constantBuffers, std::string_view debugName) {
  merian::PipelineLayoutBuilder pipelineLayoutBuilder{context};

  // input / output storage buffer descriptor set.
  merian::DescriptorSetLayoutBuilder set0Builder;
  set0Builder.add_binding_storage_buffer(); // input tensor.
  set0Builder.add_binding_storage_buffer(); // output tensor.

  for (unsigned int i = 0; i < constantBuffers; ++i) {
    set0Builder.add_binding_storage_buffer(); // constant buffers.
  }

  pipelineLayoutBuilder.add_descriptor_set_layout(
      set0Builder.build_push_descriptor_layout(context));

  pipelineLayoutBuilder.add_push_constant<glm::uvec2>();

  merian::PipelineLayoutHandle pipelineLayout =
      pipelineLayoutBuilder.build_pipeline_layout();

  merian::ShaderModuleHandle computeShaderModule;
  switch (src.lang()) {
  case ShaderLang::GLSL: {
    std::string src_name{debugName};
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

vkcnn::device::OpComputePipe::OpComputePipe(
    const merian::ContextHandle &context,
    const merian::ShaderCompilerHandle &shaderCompiler, const ShaderSource &src,
    std::span<const merian::BufferHandle> constants, glm::uvec2 tileSize,
    std::string_view debugName)
    : m_pipeline(createPipeline(context, shaderCompiler, src, constants.size(),
                                debugName)),
      m_constantBuffers(constants.begin(), constants.end()),
      m_bufferInfos(constants.size() + 2),
      m_descriptorSets(constants.size() + 2), m_tileSize(tileSize) {

  m_descriptorSets[0].setDstBinding(0);
  m_descriptorSets[0].setDescriptorType(vk::DescriptorType::eStorageBuffer);

  m_descriptorSets[1].setDstBinding(1);
  m_descriptorSets[1].setDescriptorType(vk::DescriptorType::eStorageBuffer);

  for (unsigned int i = 0; i < constants.size(); ++i) {
    m_bufferInfos[i + 2].buffer = *m_constantBuffers[i];
    m_bufferInfos[i + 2].offset = 0;
    m_bufferInfos[i + 2].range = m_constantBuffers[i]->get_size();

    m_descriptorSets[i + 2].setDstBinding(i + 2);
    m_descriptorSets[i + 2].setDescriptorType(
        vk::DescriptorType::eStorageBuffer);
    m_descriptorSets[i + 2].setBufferInfo(m_bufferInfos[i + 2]);
  }
}

void vkcnn::device::OpComputePipe::run(const merian::CommandBufferHandle &cmd,
                                       const ::vkcnn::ImageTensor &input,
                                       const ::vkcnn::ImageTensor &output) {

  m_bufferInfos[0].buffer = *input.getDeviceHandle();
  m_bufferInfos[0].offset = 0;
  m_bufferInfos[0].range = input.byteSize();

  m_bufferInfos[1].buffer = *output.getDeviceHandle();
  m_bufferInfos[1].offset = 0;
  m_bufferInfos[1].range = output.byteSize();

  cmd->bind(m_pipeline);
  m_descriptorSets[0].setBufferInfo(m_bufferInfos[0]);
  m_descriptorSets[1].setBufferInfo(m_bufferInfos[1]);
  cmd->push_descriptor_set(m_pipeline, m_descriptorSets);
  cmd->push_constant<glm::uvec2>(m_pipeline, input.imageExtent());

  glm::uvec2 workgroupCount =
      (input.imageExtent() + m_tileSize - glm::uvec2(1)) / m_tileSize;
  cmd->dispatch(workgroupCount.x, workgroupCount.y);
}

