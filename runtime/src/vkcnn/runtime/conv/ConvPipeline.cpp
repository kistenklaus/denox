#include "./ConvPipeline.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "vkcnn/runtime/tensor/SyncUse.hpp"
#include <cstring>
#include <fmt/base.h>
#include <iostream>
#include <ranges>

namespace vkcnn::runtime {

ConvPipeline::ConvPipeline(const ::merian::ContextHandle &context,
                           const ::merian::ShaderCompilerHandle &shaderCompiler,
                           const ConvShaderSource &source,
                           const FilterDeviceTensor &filterWeights)
    : m_tileSize(source.tileSize()), m_filterWeights(filterWeights)
#ifndef NDEBUG
      ,
      m_inputLayout(source.inputLayout()), m_inputType(source.inputType()),
      m_outputLayout(source.outputLayout()), m_outputType(source.outputType())
#endif
{
  assert(source.filterDesc() == filterWeights.desc());

  const ::merian::DescriptorSetLayoutHandle descriptorSet0Layout =
      ::merian::DescriptorSetLayoutBuilder()
          .add_binding_storage_buffer() // input
          .add_binding_storage_buffer() // output
          .add_binding_storage_buffer() // filter-weights.
          .build_push_descriptor_layout(context);

  std::map<std::string, std::string> defs;
  for (const auto &[def, v] : source.defines()) {
    defs[def] = v;
  }

  std::string strSrc;
  strSrc.resize(source.src().size());
  std::memcpy(strSrc.data(), source.src().data(), source.src().size_bytes());

  const ::merian::ShaderModuleHandle shader =
      shaderCompiler->compile_glsl_to_shadermodule(
          context, strSrc, source.name(), vk::ShaderStageFlagBits::eCompute, {},
          defs);

  const ::merian::PipelineLayoutHandle pipelineLayout =
      ::merian::PipelineLayoutBuilder(context)
          .add_descriptor_set_layout(descriptorSet0Layout)
          .add_push_constant<glm::uvec2>()
          .build_pipeline_layout();

  if (std::ranges::empty(source.specializationConstants())) {
    m_pipe =
        std::make_shared<::merian::ComputePipeline>(pipelineLayout, shader);
  } else {
    ::merian::SpecializationInfoBuilder specInfoBuilder;
    for (const auto &spec : source.specializationConstants()) {
      specInfoBuilder.add_entry(spec);
    }
    const ::merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();
    m_pipe = std::make_shared<::merian::ComputePipeline>(pipelineLayout, shader,
                                                         specInfo);
  }
}

void ConvPipeline::run(const ::merian::CommandBufferHandle &cmd,
                       const ActivationDeviceTensor &input,
                       const ActivationDeviceTensor &output) {
  assert(m_inputLayout == input.layout());
  assert(m_inputType == input.type());
  assert(m_outputLayout == output.layout());
  assert(m_outputType == output.type());
  cmd->bind(m_pipe);

  auto &in = input.use(cmd, SyncUseFlagBits::ComputeRead);
  auto &out = output.use(cmd, SyncUseFlagBits::ComputeWrite);
  auto &filterWeights = m_filterWeights.use(cmd, SyncUseFlagBits::ComputeRead);

  cmd->push_constant<glm::uvec2>(m_pipe, glm::uvec2(input.w(), input.h()));
  cmd->push_descriptor_set(m_pipe, in, out, filterWeights);

  glm::uvec2 workgroupCount =
      (glm::uvec2(input.w(), input.h()) + m_tileSize - glm::uvec2(1, 1)) /
      m_tileSize;
  // fmt::println("DISPATCH: ({},{})", workgroupCount.x, workgroupCount.y);
  cmd->dispatch(workgroupCount.x * workgroupCount.y);
}
} // namespace vkcnn::runtime
