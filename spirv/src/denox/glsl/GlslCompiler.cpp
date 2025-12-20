#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <memory>

namespace denox::spirv {

GlslCompiler::GlslCompiler(SpirvTools* tools, const DeviceInfo &deviceInfo,
                           SpirvDebugInfoLevel debugInfo, bool opt)
    : m_tools(tools), m_deviceInfo(deviceInfo),
      m_buildInResource(malloc(sizeof(TBuiltInResource))),
      m_debugInfo(debugInfo), m_optimize(opt) {

  auto resource = static_cast<TBuiltInResource *>(m_buildInResource);
  std::construct_at(resource);
  resource->maxLights = 32;
  resource->maxClipPlanes = 6;
  resource->maxTextureUnits = 32;
  resource->maxTextureCoords = 32;
  resource->maxVertexAttribs = 64;
  resource->maxVertexUniformComponents = 4096;
  resource->maxVaryingFloats = 64;
  resource->maxVertexTextureImageUnits = 32;
  resource->maxCombinedTextureImageUnits = 80;
  resource->maxTextureImageUnits = 32;
  resource->maxFragmentUniformComponents = 4096;
  resource->maxDrawBuffers = 32;
  resource->maxVertexUniformVectors = 128;
  resource->maxVaryingVectors = 8;
  resource->maxFragmentUniformVectors = 16;
  resource->maxVertexOutputVectors = 16;
  resource->maxFragmentInputVectors = 15;
  resource->minProgramTexelOffset = -8;
  resource->maxProgramTexelOffset = 7;
  resource->maxClipDistances = 8;
  resource->maxComputeWorkGroupCountX =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[0]);
  resource->maxComputeWorkGroupCountY =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[1]);
  resource->maxComputeWorkGroupCountZ =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupCount[2]);
  resource->maxComputeWorkGroupSizeX =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[0]);
  resource->maxComputeWorkGroupSizeY =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[1]);
  resource->maxComputeWorkGroupSizeZ =
      static_cast<int>(m_deviceInfo.limits.maxComputeWorkGroupSize[2]);
  resource->maxComputeUniformComponents = 1024;
  resource->maxComputeTextureImageUnits = 16;
  resource->maxComputeImageUniforms = 8;
  resource->maxComputeAtomicCounters = 8;
  resource->maxComputeAtomicCounterBuffers = 1;
  resource->maxVaryingComponents = 60;
  resource->maxVertexOutputComponents = 64;
  resource->maxGeometryInputComponents = 64;
  resource->maxGeometryOutputComponents = 128;
  resource->maxFragmentInputComponents = 128;
  resource->maxImageUnits = 8;
  resource->maxCombinedImageUnitsAndFragmentOutputs = 8;
  resource->maxCombinedShaderOutputResources = 8;
  resource->maxImageSamples = 0;
  resource->maxVertexImageUniforms = 0;
  resource->maxTessControlImageUniforms = 0;
  resource->maxTessEvaluationImageUniforms = 0;
  resource->maxGeometryImageUniforms = 0;
  resource->maxFragmentImageUniforms = 8;
  resource->maxCombinedImageUniforms = 8;
  resource->maxGeometryTextureImageUnits = 16;
  resource->maxGeometryOutputVertices = 256;
  resource->maxGeometryTotalOutputComponents = 1024;
  resource->maxGeometryUniformComponents = 1024;
  resource->maxGeometryVaryingComponents = 64;
  resource->maxTessControlInputComponents = 128;
  resource->maxTessControlOutputComponents = 128;
  resource->maxTessControlTextureImageUnits = 16;
  resource->maxTessControlUniformComponents = 1024;
  resource->maxTessControlTotalOutputComponents = 4096;
  resource->maxTessEvaluationInputComponents = 128;
  resource->maxTessEvaluationOutputComponents = 128;
  resource->maxTessEvaluationTextureImageUnits = 16;
  resource->maxTessEvaluationUniformComponents = 1024;
  resource->maxTessPatchComponents = 120;
  resource->maxPatchVertices = 32;
  resource->maxTessGenLevel = 64;
  resource->maxViewports = 16;
  resource->maxVertexAtomicCounters = 0;
  resource->maxTessControlAtomicCounters = 0;
  resource->maxTessEvaluationAtomicCounters = 0;
  resource->maxGeometryAtomicCounters = 0;
  resource->maxFragmentAtomicCounters = 8;
  resource->maxCombinedAtomicCounters = 8;
  resource->maxAtomicCounterBindings = 1;
  resource->maxVertexAtomicCounterBuffers = 0;
  resource->maxTessControlAtomicCounterBuffers = 0;
  resource->maxTessEvaluationAtomicCounterBuffers = 0;
  resource->maxGeometryAtomicCounterBuffers = 0;
  resource->maxFragmentAtomicCounterBuffers = 1;
  resource->maxCombinedAtomicCounterBuffers = 1;
  resource->maxAtomicCounterBufferSize = 16384;
  resource->maxTransformFeedbackBuffers = 4;
  resource->maxTransformFeedbackInterleavedComponents = 64;
  resource->maxCullDistances = 8;
  resource->maxCombinedClipAndCullDistances = 8;
  resource->maxSamples = 4;
  resource->maxMeshOutputVerticesNV = 256;
  resource->maxMeshOutputPrimitivesNV = 512;
  resource->maxMeshWorkGroupSizeX_NV = 32;
  resource->maxMeshWorkGroupSizeY_NV = 1;
  resource->maxMeshWorkGroupSizeZ_NV = 1;
  resource->maxTaskWorkGroupSizeX_NV = 32;
  resource->maxTaskWorkGroupSizeY_NV = 1;
  resource->maxTaskWorkGroupSizeZ_NV = 1;
  resource->maxMeshViewCountNV = 4;
  resource->limits.nonInductiveForLoops = true;
  resource->limits.whileLoops = true;
  resource->limits.doWhileLoops = true;
  resource->limits.generalUniformIndexing = true;
  resource->limits.generalAttributeMatrixVectorIndexing = true;
  resource->limits.generalVaryingIndexing = true;
  resource->limits.generalSamplerIndexing = true;
  resource->limits.generalVariableIndexing = true;
  resource->limits.generalConstantMatrixVectorIndexing = true;
}

GlslCompiler::~GlslCompiler() {
  free(m_buildInResource);
  m_buildInResource = nullptr;
}

GlslCompilerInstance GlslCompiler::read(io::Path sourcePath) {
  io::File file = io::File::open(sourcePath, io::File::OpenMode::Read);

  memory::vector<std::byte> glslSource(file.size());
  file.read_exact(glslSource);

  return GlslCompilerInstance(this, std::move(glslSource), std::move(sourcePath));
}

} // namespace denox::spirv
