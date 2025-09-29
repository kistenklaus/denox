#pragma once

#include "io/fs/Path.hpp"
#include "memory/container/string.hpp"
#include "memory/container/string_view.hpp"
#include "shaders/compiler/CompilationResult.hpp"
#include "shaders/compiler/ShaderBinary.hpp"
#include <concepts>
#include <fmt/format.h>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>

namespace denox::compiler {

class GlslCompiler;

class GlslCompilerInstance {
public:
  friend class GlslCompiler;

  template <typename T>
    requires(fmt::is_formattable<T>::value && !std::same_as<T, memory::string_view> &&
             !std::same_as<T, bool>)
  void define(std::string_view name, const T &value) {
    m_preamble.append(fmt::format("#define {} ({})\n", name, value));
  }

  void define(std::string_view name, memory::string_view value) {
    m_preamble.append(fmt::format("#define {} {}\n", name, value));
  }

  void define(std::string_view name, bool value) {
    m_preamble.append(fmt::format("#define {} ({})\n", name, value ? 0 : 1));
  }

  void enableDenoxPreprocessor() { m_denoxPreprocessor = true; }

  CompilationResult compile();

private:
  GlslCompilerInstance(GlslCompiler *compiler, glslang::TShader shader,
                       memory::vector<std::byte> src,
                       io::Path sourcePath)
      : m_compiler(compiler), m_shader(std::move(shader)),
        m_src(std::move(src)), m_sourcePath(std::move(sourcePath)) {}

private:
  GlslCompiler *m_compiler;
  glslang::TShader m_shader;
  memory::string m_preamble;
  bool m_denoxPreprocessor;
  memory::vector<std::byte> m_src;
  io::Path m_sourcePath;
};

} // namespace denox::compiler
