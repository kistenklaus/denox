#pragma once

#include "shader/GlslCompilerInstance.hpp"
#include <glslang/Include/ResourceLimits.h>

namespace denox::glsl {


class GlslCompiler {
public:
  friend class GlslCompilerInstance;
  GlslCompiler();

  GlslCompilerInstance read(std::string sourcePath);

private:
  TBuiltInResource m_buildInResource;
};

} // namespace denox::compiler
