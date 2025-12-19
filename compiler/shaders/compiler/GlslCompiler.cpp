#include "shaders/compiler/GlslCompiler.hpp"
#include "denox/io/fs/File.hpp"
#include "shaders/compiler/GlslCompilerInstance.hpp"
#include <glslang/Public/ShaderLang.h>

namespace denox::compiler {

GlslCompiler::GlslCompiler(const Options &options)
    : m_deviceInfo(options.deviceInfo), m_buildInResource(),
      m_debugInfo(options.shaderDebugInfo), m_optimize(options.optimizeSpirv) {
  m_buildInResource.maxLights = 32;
  m_buildInResource.maxClipPlanes = 6;
  m_buildInResource.maxTextureUnits = 32;
  m_buildInResource.maxTextureCoords = 32;
  m_buildInResource.maxVertexAttribs = 64;
  m_buildInResource.maxVertexUniformComponents = 4096;
  m_buildInResource.maxVaryingFloats = 64;
  m_buildInResource.maxVertexTextureImageUnits = 32;
  m_buildInResource.maxCombinedTextureImageUnits = 80;
  m_buildInResource.maxTextureImageUnits = 32;
  m_buildInResource.maxFragmentUniformComponents = 4096;
  m_buildInResource.maxDrawBuffers = 32;
  m_buildInResource.maxVertexUniformVectors = 128;
  m_buildInResource.maxVaryingVectors = 8;
  m_buildInResource.maxFragmentUniformVectors = 16;
  m_buildInResource.maxVertexOutputVectors = 16;
  m_buildInResource.maxFragmentInputVectors = 15;
  m_buildInResource.minProgramTexelOffset = -8;
  m_buildInResource.maxProgramTexelOffset = 7;
  m_buildInResource.maxClipDistances = 8;
  m_buildInResource.maxComputeWorkGroupCountX =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[0]);
  m_buildInResource.maxComputeWorkGroupCountY =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[1]);
  m_buildInResource.maxComputeWorkGroupCountZ =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[2]);
  m_buildInResource.maxComputeWorkGroupSizeX =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[0]);
  m_buildInResource.maxComputeWorkGroupSizeY =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[1]);
  m_buildInResource.maxComputeWorkGroupSizeZ =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[2]);
  m_buildInResource.maxComputeUniformComponents = 1024;
  m_buildInResource.maxComputeTextureImageUnits = 16;
  m_buildInResource.maxComputeImageUniforms = 8;
  m_buildInResource.maxComputeAtomicCounters = 8;
  m_buildInResource.maxComputeAtomicCounterBuffers = 1;
  m_buildInResource.maxVaryingComponents = 60;
  m_buildInResource.maxVertexOutputComponents = 64;
  m_buildInResource.maxGeometryInputComponents = 64;
  m_buildInResource.maxGeometryOutputComponents = 128;
  m_buildInResource.maxFragmentInputComponents = 128;
  m_buildInResource.maxImageUnits = 8;
  m_buildInResource.maxCombinedImageUnitsAndFragmentOutputs = 8;
  m_buildInResource.maxCombinedShaderOutputResources = 8;
  m_buildInResource.maxImageSamples = 0;
  m_buildInResource.maxVertexImageUniforms = 0;
  m_buildInResource.maxTessControlImageUniforms = 0;
  m_buildInResource.maxTessEvaluationImageUniforms = 0;
  m_buildInResource.maxGeometryImageUniforms = 0;
  m_buildInResource.maxFragmentImageUniforms = 8;
  m_buildInResource.maxCombinedImageUniforms = 8;
  m_buildInResource.maxGeometryTextureImageUnits = 16;
  m_buildInResource.maxGeometryOutputVertices = 256;
  m_buildInResource.maxGeometryTotalOutputComponents = 1024;
  m_buildInResource.maxGeometryUniformComponents = 1024;
  m_buildInResource.maxGeometryVaryingComponents = 64;
  m_buildInResource.maxTessControlInputComponents = 128;
  m_buildInResource.maxTessControlOutputComponents = 128;
  m_buildInResource.maxTessControlTextureImageUnits = 16;
  m_buildInResource.maxTessControlUniformComponents = 1024;
  m_buildInResource.maxTessControlTotalOutputComponents = 4096;
  m_buildInResource.maxTessEvaluationInputComponents = 128;
  m_buildInResource.maxTessEvaluationOutputComponents = 128;
  m_buildInResource.maxTessEvaluationTextureImageUnits = 16;
  m_buildInResource.maxTessEvaluationUniformComponents = 1024;
  m_buildInResource.maxTessPatchComponents = 120;
  m_buildInResource.maxPatchVertices = 32;
  m_buildInResource.maxTessGenLevel = 64;
  m_buildInResource.maxViewports = 16;
  m_buildInResource.maxVertexAtomicCounters = 0;
  m_buildInResource.maxTessControlAtomicCounters = 0;
  m_buildInResource.maxTessEvaluationAtomicCounters = 0;
  m_buildInResource.maxGeometryAtomicCounters = 0;
  m_buildInResource.maxFragmentAtomicCounters = 8;
  m_buildInResource.maxCombinedAtomicCounters = 8;
  m_buildInResource.maxAtomicCounterBindings = 1;
  m_buildInResource.maxVertexAtomicCounterBuffers = 0;
  m_buildInResource.maxTessControlAtomicCounterBuffers = 0;
  m_buildInResource.maxTessEvaluationAtomicCounterBuffers = 0;
  m_buildInResource.maxGeometryAtomicCounterBuffers = 0;
  m_buildInResource.maxFragmentAtomicCounterBuffers = 1;
  m_buildInResource.maxCombinedAtomicCounterBuffers = 1;
  m_buildInResource.maxAtomicCounterBufferSize = 16384;
  m_buildInResource.maxTransformFeedbackBuffers = 4;
  m_buildInResource.maxTransformFeedbackInterleavedComponents = 64;
  m_buildInResource.maxCullDistances = 8;
  m_buildInResource.maxCombinedClipAndCullDistances = 8;
  m_buildInResource.maxSamples = 4;
  m_buildInResource.maxMeshOutputVerticesNV = 256;
  m_buildInResource.maxMeshOutputPrimitivesNV = 512;
  m_buildInResource.maxMeshWorkGroupSizeX_NV = 32;
  m_buildInResource.maxMeshWorkGroupSizeY_NV = 1;
  m_buildInResource.maxMeshWorkGroupSizeZ_NV = 1;
  m_buildInResource.maxTaskWorkGroupSizeX_NV = 32;
  m_buildInResource.maxTaskWorkGroupSizeY_NV = 1;
  m_buildInResource.maxTaskWorkGroupSizeZ_NV = 1;
  m_buildInResource.maxMeshViewCountNV = 4;
  m_buildInResource.limits.nonInductiveForLoops = true;
  m_buildInResource.limits.whileLoops = true;
  m_buildInResource.limits.doWhileLoops = true;
  m_buildInResource.limits.generalUniformIndexing = true;
  m_buildInResource.limits.generalAttributeMatrixVectorIndexing = true;
  m_buildInResource.limits.generalVaryingIndexing = true;
  m_buildInResource.limits.generalSamplerIndexing = true;
  m_buildInResource.limits.generalVariableIndexing = true;
  m_buildInResource.limits.generalConstantMatrixVectorIndexing = true;
}

GlslCompilerInstance GlslCompiler::read(io::Path sourcePath) {
  io::File file = io::File::open(sourcePath, io::File::OpenMode::Read);

  memory::vector<std::byte> glslSource(file.size());
  file.read_exact(glslSource);

  auto shader = std::make_unique<glslang::TShader>(EShLangCompute);
  memory::string source_path_str = sourcePath.str();
  shader->setSourceFile(source_path_str.data());

  shader->setEnvInput(glslang::EShSourceGlsl, EShLangCompute,
                     glslang::EShClientVulkan, 100);

  switch (m_deviceInfo.apiVersion) {
  case ApiVersion::VULKAN_1_0:
    shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    break;
  case ApiVersion::VULKAN_1_1:
    shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    break;
  case ApiVersion::VULKAN_1_2:
    shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
    shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
    break;
  case ApiVersion::VULKAN_1_3:
    shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    break;
  case ApiVersion::VULKAN_1_4:
    shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_4);
    shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    break;
  }

  return GlslCompilerInstance(this, std::move(shader), std::move(glslSource),
                              std::move(sourcePath));
}

} // namespace denox::compiler
