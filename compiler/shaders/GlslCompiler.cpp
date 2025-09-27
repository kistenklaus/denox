#include "shaders/GlslCompiler.hpp"

#include "glslang/Public/ShaderLang.h"
#include "shaders/global_glslang_runtime.hpp"
#include <glslang/Include/ResourceLimits.h>

namespace denox::compiler {

struct CompilerControlBlock {
  TBuiltInResource resource;
};

static TBuiltInResource GetDefaultResource() {
  TBuiltInResource r = {};
  r.maxDrawBuffers = 32;
  r.maxImageUnits = 32;
  r.maxCombinedTextureImageUnits = 80;
  r.maxTextureImageUnits = 32;
  r.maxVertexTextureImageUnits = 32;
  r.maxCombinedShaderOutputResources = 32;
  r.maxVertexOutputComponents = 64;
  r.maxFragmentInputComponents = 128;
  r.maxVertexUniformComponents = 4096;
  r.maxFragmentUniformComponents = 4096;
  r.maxComputeWorkGroupCountX = 65535;
  r.maxComputeWorkGroupCountY = 65535;
  r.maxComputeWorkGroupCountZ = 65535;
  r.maxComputeWorkGroupSizeX = 1024;
  r.maxComputeWorkGroupSizeY = 1024;
  r.maxComputeWorkGroupSizeZ = 64;
  r.maxClipDistances = 8;
  r.maxCullDistances = 8;
  r.limits.nonInductiveForLoops = true;
  r.limits.whileLoops = true;
  r.limits.doWhileLoops = true;
  r.limits.generalUniformIndexing = true;
  r.limits.generalAttributeMatrixVectorIndexing = true;
  r.limits.generalConstantMatrixVectorIndexing = true;
  r.limits.generalSamplerIndexing = true;
  r.limits.generalVariableIndexing = true;
  return r;
}

GlslCompiler::GlslCompiler() : m_self(new CompilerControlBlock()) {
  global_glslang_runtime::ensure_initialized();
  auto self = static_cast<CompilerControlBlock *>(m_self);
  new (self) CompilerControlBlock(GetDefaultResource());
}

GlslCompiler::~GlslCompiler() {
  delete static_cast<CompilerControlBlock *>(m_self);
}

GlslCompilerInstance GlslCompiler::invoke([[maybe_unused]] io::Path source_path) {
  return GlslCompilerInstance{};
}

} // namespace denox::compiler
