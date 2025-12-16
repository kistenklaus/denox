#include "shader/GlslCompiler.hpp"
#include "shader/GlslCompilerInstance.hpp"
#include <fstream>
#include <glslang/Public/ShaderLang.h>

namespace denox::glsl {

GlslCompiler::GlslCompiler() : m_buildInResource() {
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
  m_buildInResource.maxComputeWorkGroupCountX = 65535;
  m_buildInResource.maxComputeWorkGroupCountY = 65535;
  m_buildInResource.maxComputeWorkGroupCountZ = 65535;
  m_buildInResource.maxComputeWorkGroupSizeX = 1024;
  m_buildInResource.maxComputeWorkGroupSizeY = 1024;
  m_buildInResource.maxComputeWorkGroupSizeZ = 64;
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

std::vector<std::byte> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  const std::streamsize size = file.tellg();
  if (size < 0) {
    throw std::runtime_error("Failed to get file size: " + path);
  }

  std::vector<std::byte> buffer(static_cast<size_t>(size));

  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(buffer.data()), size);

  return buffer;
}

GlslCompilerInstance GlslCompiler::read(std::string sourcePath) {

  std::vector<std::byte> glslSource = read_file_bytes(sourcePath);

  auto shader = std::make_unique<glslang::TShader>(EShLangCompute);
  shader->setSourceFile(sourcePath.data());

  shader->setEnvInput(glslang::EShSourceGlsl, EShLangCompute,
                      glslang::EShClientVulkan, 100);
  shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_4);
  shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

  return GlslCompilerInstance(this, std::move(shader), std::move(glslSource),
                              std::move(sourcePath));
}

} // namespace denox::compiler
