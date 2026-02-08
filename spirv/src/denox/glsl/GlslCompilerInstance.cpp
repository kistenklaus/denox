#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/glsl/GlslPreprocessor.hpp"
#include "denox/memory/container/vector.hpp"
#include <glslang/Include/ResourceLimits.h>
#include <glslang/MachineIndependent/Versions.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/SPIRV/Logger.h>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace denox::spirv {

CompilationResult GlslCompilerInstance::compile() {
  std::lock_guard lck{*m_mutex};

  ::glslang::TShader shader(EShLangCompute);
  shader.setSourceFile(m_sourcePath.cstr());
  shader.setEnvInput(::glslang::EShSourceGlsl, EShLangCompute,
                     ::glslang::EShClientVulkan, 100);

  switch (m_compiler->m_deviceInfo.apiVersion) {
  case ApiVersion::VULKAN_1_0:
    shader.setEnvClient(::glslang::EShClientVulkan,
                        ::glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(::glslang::EShTargetSpv, ::glslang::EShTargetSpv_1_0);
    break;
  case ApiVersion::VULKAN_1_1:
    shader.setEnvClient(::glslang::EShClientVulkan,
                        ::glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(::glslang::EShTargetSpv, ::glslang::EShTargetSpv_1_3);
    break;
  case ApiVersion::VULKAN_1_2:
    shader.setEnvClient(::glslang::EShClientVulkan,
                        ::glslang::EShTargetVulkan_1_2);
    shader.setEnvTarget(::glslang::EShTargetSpv, ::glslang::EShTargetSpv_1_5);
    break;
  case ApiVersion::VULKAN_1_3:
    shader.setEnvClient(::glslang::EShClientVulkan,
                        ::glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(::glslang::EShTargetSpv, ::glslang::EShTargetSpv_1_6);
    break;
  case ApiVersion::VULKAN_1_4:
    shader.setEnvClient(::glslang::EShClientVulkan,
                        ::glslang::EShTargetVulkan_1_4);
    shader.setEnvTarget(::glslang::EShTargetSpv, ::glslang::EShTargetSpv_1_6);
    break;
  default:
    diag::unreachable();
  }

  const char *preambleCStr = m_preamble.data();
  shader.setPreamble(preambleCStr);

  GlslPreprocessor preprocessor;
  memory::string preprocessed = preprocessor.preprocess(memory::string_view{
      reinterpret_cast<const char *>(m_src.data()), m_src.size()});
  // DENOX_WARN("FINAL GLSL: {}\n{}{}", m_sourcePath, m_preamble, preprocessed);

  const char *srcPtr = preprocessed.c_str();
  shader.setStrings(&srcPtr, 1);

  const TBuiltInResource *buildInResource =
      static_cast<const TBuiltInResource *>(m_compiler->m_buildInResource);

  EShMessages messages = static_cast<EShMessages>(
      EShMsgDefault | EShMsgEnhanced | EShMsgSpvRules | EShMsgVulkanRules);
  switch (m_compiler->m_debugInfo) {
  case SpirvDebugInfoLevel::Strip:
    shader.setDebugInfo(false);
    break;
  case SpirvDebugInfoLevel::Enable:
  case SpirvDebugInfoLevel::ForceNonSemanticDebugInfo:
    messages = static_cast<EShMessages>(messages | EShMsgDebugInfo);
    shader.setDebugInfo(true);
    break;
  default:
    diag::unreachable();
  }

  //  =============::glslang-FRONTEND-STAGE==============
  if (!shader.parse(buildInResource, 450, false, messages)) {
    return CompilationError{CompilationStage::GlslangParse,
                            fmt::format("[::glslang-parse]: {}\n{}",
                                        m_sourcePath.str(),
                                        shader.getInfoLog())};
  }

  {
    const char *w1 = shader.getInfoLog();
    if (w1 && *w1) { // <- check if empty string.
      DENOX_WARN("[::glslang-parse]: {}\n{}", m_sourcePath.str(), w1);
    }
  }

  ::glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    return CompilationError{CompilationStage::GlslangLink,
                            memory::string(program.getInfoLog())};
  }
  {
    const char *w1 = program.getInfoLog();
    if (w1 && *w1) {
      DENOX_WARN("[::glslang-link]: {}\n{}", m_sourcePath.str(), w1);
    }
  }

  ::glslang::TIntermediate *intermediate =
      program.getIntermediate(EShLangCompute);
  if (!intermediate) {
    return CompilationError{CompilationStage::GlslangIntermediate,
                            memory::string(program.getInfoLog())};
  }

  spv::SpvBuildLogger logger;
  ::glslang::SpvOptions spvOptions;
  bool enableDebugInfo;
  bool enableNonSemanticDebugInfo;
  switch (m_compiler->m_debugInfo) {
  case SpirvDebugInfoLevel::Strip:
    enableDebugInfo = false;
    enableNonSemanticDebugInfo = false;
    break;
  case SpirvDebugInfoLevel::Enable:
    enableDebugInfo = true;
    enableNonSemanticDebugInfo = false;
    break;
  case SpirvDebugInfoLevel::ForceNonSemanticDebugInfo:
    enableDebugInfo = true;
    enableNonSemanticDebugInfo = true;
    break;
  default:
    diag::unreachable();
  };

  spvOptions.generateDebugInfo = enableDebugInfo;
  spvOptions.stripDebugInfo = !enableDebugInfo;
  spvOptions.disableOptimizer = !m_compiler->m_optimize;
  spvOptions.optimizeSize = false;
  spvOptions.disassemble = false;
  spvOptions.validate = false;
  spvOptions.emitNonSemanticShaderDebugInfo = enableNonSemanticDebugInfo;
  spvOptions.emitNonSemanticShaderDebugSource = enableNonSemanticDebugInfo;
  spvOptions.compileOnly = false;
  spvOptions.optimizerAllowExpandedIDBound = m_compiler->m_optimize;

  memory::vector<std::uint32_t> spirv;
  ::glslang::GlslangToSpv(*intermediate, spirv, &logger, &spvOptions);
  memory::string glslangSpvLog = logger.getAllMessages();
  if (!glslangSpvLog.empty()) {
    DENOX_WARN("[::glslang-spirv]: {}\n{}", m_sourcePath.str(), glslangSpvLog);
  }

  SpirvBinary binary{spirv};
  if (!m_compiler->m_tools->validate(binary)) {
    return CompilationError{
        .stage = CompilationStage::SpirvToolsVal,
        .msg = m_compiler->m_tools->get_error_msg(),
    };
  }
  if (m_compiler->m_optimize) {
    if (!m_compiler->m_tools->optimize(binary)) {
      return CompilationError{
          .stage = CompilationStage::SpirvToolsOpt,
          .msg = m_compiler->m_tools->get_error_msg(),
      };
    }
  }

  return binary;
}

SHA256 GlslCompilerInstance::fast_sha256() const {
  SHA256Builder hasher = m_compiler->m_shaCache.cachedSHA(m_sourcePath, m_src);
  hasher.update(std::span{reinterpret_cast<const uint8_t *>(m_preamble.data()),
                          m_preamble.size()});
  return hasher.finalize();
}

} // namespace denox::spirv
