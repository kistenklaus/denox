#include "shader/GlslCompilerInstance.hpp"
#include "shader/CompilationError.hpp"
#include "shader/GlslCompiler.hpp"
#include <cstring>
#include <glslang/MachineIndependent/Versions.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/SPIRV/Logger.h>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <fmt/printf.h>

namespace denox::glsl {

CompilationResult GlslCompilerInstance::compile() {

  const char *preambleCStr = m_preamble.data();

  unsigned int vulkanApiVersionOrd = 4;

  m_shader->setPreamble(preambleCStr);

  std::string src(m_src.size() + 1, '\0');
  std::memcpy(src.data(), m_src.data(), m_src.size());
  const char *srcPtr = src.data();

  m_shader->setStrings(&srcPtr, 1);

  const TBuiltInResource &buildInResource = m_compiler->m_buildInResource;

  EShMessages messages = static_cast<EShMessages>(
      EShMsgDefault | EShMsgEnhanced | EShMsgSpvRules | EShMsgVulkanRules);
  m_shader->setDebugInfo(false);

  //  =============GLSLANG-FRONTEND-STAGE==============
  if (!m_shader->parse(&buildInResource, 450, false, messages)) {
    return CompilationError{
        CompilationStage::GlslangParse,
        fmt::format("\x1B[33m[glslang-parse]\x1B[0m: {}\n{}", m_sourcePath,
                    m_shader->getInfoLog())};
  }
  {
    const char *w1 = m_shader->getInfoLog();
    if (w1 && *w1) { // <- check if empty string.
      fmt::println("\x1B[33m[glslang-parse]\x1B[0m: {}\n{}", m_sourcePath, w1);
    }
  }

  glslang::TProgram program;
  program.addShader(m_shader.get());
  if (!program.link(messages)) {
    return CompilationError{CompilationStage::GlslangLink,
                            std::string(program.getInfoLog())};
  }
  {
    const char *w1 = program.getInfoLog();
    if (w1 && *w1) {
      fmt::println("\x1B[33m[glslang-link]\x1B[0m: {}\n{}", m_sourcePath, w1);
    }
  }

  glslang::TIntermediate *intermediate =
      program.getIntermediate(EShLangCompute);
  if (!intermediate) {
    return CompilationError{CompilationStage::GlslangIntermediate,
                            std::string(program.getInfoLog())};
  }

  spv::SpvBuildLogger logger;
  glslang::SpvOptions spvOptions;
  bool enableDebugInfo = false;
  bool enableNonSemanticDebugInfo = false;

  spvOptions.generateDebugInfo = enableDebugInfo;
  spvOptions.stripDebugInfo = !enableDebugInfo;
  spvOptions.disableOptimizer = false;
  spvOptions.optimizeSize = false;
  spvOptions.disassemble = false;
  spvOptions.validate = false;
  spvOptions.emitNonSemanticShaderDebugInfo = enableNonSemanticDebugInfo;
  spvOptions.emitNonSemanticShaderDebugSource = enableNonSemanticDebugInfo;
  spvOptions.compileOnly = false;
  spvOptions.optimizerAllowExpandedIDBound = true;

  std::vector<std::uint32_t> spirv;
  glslang::GlslangToSpv(*intermediate, spirv, &logger, &spvOptions);
  std::string glslangSpvLog = logger.getAllMessages();
  if (!glslangSpvLog.empty()) {
    fmt::println("\x1B[33m[glslang-spirv]\x1B[0m: {}\n{}", m_sourcePath, glslangSpvLog);
  }

  //  =============SPIRV-TOOLS-VALIDATION==============
  spv_target_env spvTargetEnv = SPV_ENV_VULKAN_1_4;

  spvtools::SpirvTools tools(spvTargetEnv);
  std::string spvLog;
  const char *spvStage = "spv-val";
  auto spvLogger = [&](spv_message_level_t level, const char *source,
                       const spv_position_t &p, const char *msg) {
    // ANSI colors (toggle if needed)
    static const bool use_color = true;
    const char *R = use_color ? "\x1b[0m" : "";
    const char *B = use_color ? "\x1b[1m" : "";
    const char *RED = use_color ? "\x1b[31m" : "";
    const char *YEL = use_color ? "\x1b[33m" : "";
    const char *CYN = use_color ? "\x1b[36m" : "";
    const char *MAG = use_color ? "\x1b[35m" : "";
    const char *GRY = use_color ? "\x1b[90m" : "";

    // Map level → label/color
    const char *label = "message";
    const char *col = GRY;
    switch (level) {
    case SPV_MSG_FATAL:
      label = "fatal error";
      col = RED;
      break;
    case SPV_MSG_INTERNAL_ERROR:
      label = "internal error";
      col = MAG;
      break;
    case SPV_MSG_ERROR:
      label = "error";
      col = RED;
      break;
    case SPV_MSG_WARNING:
      label = "warning";
      col = YEL;
      break;
    case SPV_MSG_INFO:
      label = "info";
      col = CYN;
      break;
    case SPV_MSG_DEBUG:
      label = "debug";
      col = GRY;
      break;
    default:
      label = "message";
      col = GRY;
      break;
    }

    if (!source)
      source = spvStage;
    if (!msg)
      msg = "(no message)";

    // Trim trailing whitespace/newlines (keep it compact like compilers)
    std::string_view m{msg};
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' ||
                          m.back() == ' ' || m.back() == '\t'))
      m.remove_suffix(1);

    // <source>:<line>:<col>: <severity>: <message>
    fmt::format_to(std::back_inserter(spvLog), "{}{}{}", B, source, R);
    if (p.line || p.column)
      fmt::format_to(std::back_inserter(spvLog), ":{}:{}", p.line ? p.line : 0,
                     p.column ? p.column : 0);
    fmt::format_to(std::back_inserter(spvLog), ": {}{}{}: {}\n", col, label, R,
                   m);

    // Optional extra context (word index) in dim text
    if (p.index)
      fmt::format_to(std::back_inserter(spvLog), "{}  note: word-index {}{}\n",
                     GRY, p.index, R);
  };
  tools.SetMessageConsumer(spvLogger);

  spvtools::ValidatorOptions valOptions;

  // Only if you want extra guardrails; otherwise omit these:
  valOptions.SetUniversalLimit(spv_validator_limit_max_struct_members, 1024);
  valOptions.SetUniversalLimit(spv_validator_limit_max_struct_depth, 64);
  valOptions.SetUniversalLimit(spv_validator_limit_max_local_variables, 8192);
  valOptions.SetUniversalLimit(spv_validator_limit_max_global_variables, 4096);
  valOptions.SetUniversalLimit(spv_validator_limit_max_switch_branches, 16384);
  valOptions.SetUniversalLimit(spv_validator_limit_max_function_args, 64);
  valOptions.SetUniversalLimit(
      spv_validator_limit_max_control_flow_nesting_depth, 128);
  valOptions.SetUniversalLimit(spv_validator_limit_max_access_chain_indexes,
                               255);
  // Be careful with this one if you also run optimizers:
  valOptions.SetUniversalLimit(spv_validator_limit_max_id_bound, 1u << 24);

  valOptions.SetRelaxBlockLayout(vulkanApiVersionOrd >= 1);
  valOptions.SetRelaxStructStore(false);
  valOptions.SetUniformBufferStandardLayout(false);
  valOptions.SetScalarBlockLayout(false);
  valOptions.SetWorkgroupScalarBlockLayout(false);
  valOptions.SetSkipBlockLayout(false);
  valOptions.SetAllowLocalSizeId(false); // <- spec constants.
  valOptions.SetAllowOffsetTextureOperand(false);
  valOptions.SetAllowVulkan32BitBitwise(false);
  valOptions.SetRelaxLogicalPointer(false);
  valOptions.SetBeforeHlslLegalization(false);
  valOptions.SetFriendlyNames(true);

  bool valOk = tools.Validate(spirv.data(), spirv.size(), valOptions);
  if (!valOk) {
    return CompilationError{
        CompilationStage::SpirvToolsVal,
        std::string(spvLog),
    };
  }

  if (!spvLog.empty()) {
    fmt::println("\x1B[33m[spv-val]\x1B[0m: {}\n{}", m_sourcePath, spvLog);
  }
  spvLog.clear();

  spvtools::OptimizerOptions optOptions;
  optOptions.set_run_validator(true);
  optOptions.set_preserve_bindings(true);
  optOptions.set_validator_options(valOptions);
  optOptions.set_preserve_spec_constants(true);

  spvtools::Optimizer spvOptimizer(spvTargetEnv);
  spvLog.clear();
  spvStage = "spv-opt";
  spvOptimizer.SetMessageConsumer(spvLogger);
  spvOptimizer.SetTargetEnv(spvTargetEnv);
  spvOptimizer.SetValidateAfterAll(true);

  spvOptimizer.RegisterPerformancePasses(/*preserve_interface=*/true);

  std::vector<std::uint32_t> current = std::move(spirv);
  std::vector<std::uint32_t> next;

  bool optimizeUntilConvergence = false;
  for (int iter = 0; iter < (optimizeUntilConvergence ? 10 : 1); ++iter) {
    next.clear();
    bool ok =
        spvOptimizer.Run(current.data(), current.size(), &next, optOptions);
    if (!ok) {
      return CompilationError{
          CompilationStage::SpirvToolsOpt,
          std::string(spvLog),
      };
    }

    if (next == current) {
      break; // converged → stop
    }
    current.swap(next); // keep improving
  }

  if (!spvLog.empty()) {
    fmt::println("[spv-opt]: {}", spvLog);
  }
  spvLog.clear();
  return ShaderBinary{
      .spv = std::move(current),
  };
}
} // namespace denox::compiler
