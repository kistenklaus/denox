#include "./ActivationPipeline.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <cassert>
#include <cstring>
#include <glm/fwd.hpp>
#include <ranges>

namespace vkcnn::runtime {

ActivationPipeline::ActivationPipeline(const ::merian::ContextHandle &context,
                           const ::merian::ShaderCompilerHandle &shaderCompiler,
                           const ActivationShaderSource &source)
    : m_tileSize(source.tileSize()) {

  ::merian::DescriptorSetLayoutBuilder descriptorSet0Builder;
  descriptorSet0Builder.add_binding_storage_buffer(); // input
  descriptorSet0Builder.add_binding_storage_buffer(); // output

  ::merian::DescriptorSetLayoutHandle descriptorSet0Layout =
      descriptorSet0Builder.build_push_descriptor_layout(context);
  std::map<std::string, std::string> defs;

  for (const auto &[def, v] : source.defines()) {
    defs[def] = v;
  }

  std::string strSrc;
  strSrc.resize(source.src().size());
  std::memcpy(strSrc.data(), source.src().data(), source.src().size_bytes());

#ifndef NDEBUG
  ::merian::ShaderModuleHandle shader =
      shaderCompiler->compile_glsl_to_shadermodule(
          context, strSrc, source.debugInfo().name,
          vk::ShaderStageFlagBits::eCompute, {}, defs);
#else
  ::merian::ShaderModuleHandle shader =
      shaderCompiler->compile_glsl_to_shadermodule(
          context, strSrc, "copy-no-debug-info",
          vk::ShaderStageFlagBits::eCompute, {}, defs);
#endif
  const ::merian::PipelineLayoutHandle pipelineLayout =
      ::merian::PipelineLayoutBuilder(context)
          .add_descriptor_set_layout(descriptorSet0Layout)
          .add_push_constant<glm::uvec2>()
          .build_pipeline_layout();

  if (std::ranges::empty(source.specConstants())) {
    m_pipe =
        std::make_shared<::merian::ComputePipeline>(pipelineLayout, shader);
  } else {
    ::merian::SpecializationInfoBuilder specInfoBuilder;
    for (const auto &spec : source.specConstants()) {
      specInfoBuilder.add_entry(spec);
    }
    const ::merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();
    m_pipe = std::make_shared<::merian::ComputePipeline>(pipelineLayout, shader,
                                                         specInfo);
  }
}

void ActivationPipeline::run(const ::merian::CommandBufferHandle &cmd,
                       const ActivationDeviceTensor &input,
                       const ActivationDeviceTensor &output) {
  assert(input.w() == output.w());
  assert(input.h() == output.h());

  cmd->bind(m_pipe);

  auto &in = input.use(cmd, SyncUseFlagBits::ComputeRead);
  auto &out = output.use(cmd, SyncUseFlagBits::ComputeWrite);

  struct PushConstant {
    std::uint32_t W;
    std::uint32_t H;
  };
  cmd->push_constant<PushConstant>(m_pipe, PushConstant(input.w(), input.h()));
  cmd->push_descriptor_set(m_pipe, in, out);

  glm::uvec3 workgroupCount = (glm::uvec3(input.c(), input.w(), input.h()) +
                               m_tileSize - glm::uvec3(1, 1, 1)) /
                              m_tileSize;
#ifndef NDEBUG
  fmt::println("DISPATCH: ({},{},{})", workgroupCount.x, workgroupCount.y,
               workgroupCount.z);
#endif
  cmd->dispatch(workgroupCount.x, workgroupCount.y, workgroupCount.z);
}

}
