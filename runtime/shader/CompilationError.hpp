#pragma once

#include <string>
namespace denox::glsl {

enum class CompilationStage {
  GlslangPreprocess,
  GlslangParse,
  GlslangLink,
  GlslangIntermediate,
  SpirvToolsVal,
  SpirvToolsOpt,
};

struct CompilationError {
  CompilationStage stage;
  std::string msg;
};

}
