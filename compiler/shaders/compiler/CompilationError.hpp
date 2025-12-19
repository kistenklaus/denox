#pragma once

#include "denox/memory/container/string.hpp"
namespace denox::compiler {

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
  memory::string msg;
};

}
