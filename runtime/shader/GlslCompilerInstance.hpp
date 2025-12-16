#pragma once

#include "shader/CompilationResult.hpp"
#include <concepts>
#include <fmt/format.h>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <memory>

namespace denox::glsl {

class GlslCompiler;

class GlslCompilerInstance {
public:
  friend class GlslCompiler;

  template <typename T>
    requires(fmt::is_formattable<T>::value &&
             !std::same_as<T, std::string_view> &&
             !std::same_as<T, std::string> && !std::same_as<T, const char *> &&
             !std::same_as<T, bool>)
  void define(std::string_view name, const T &value) {
    m_preamble.append(fmt::format("#define {} ({})\n", name, value));
  }

  void define(std::string_view name, std::string_view value) {
    m_preamble.append(fmt::format("#define {} {}\n", name, value));
  }

  void define(std::string_view name, std::string value) {
    m_preamble.append(fmt::format("#define {} {}\n", name, value));
  }

  void define(std::string_view name, const char *value) {
    m_preamble.append(fmt::format("#define {} {}\n", name, value));
  }

  void define(std::string_view name, bool value) {
    m_preamble.append(fmt::format("#define {} ({})\n", name, value ? 0 : 1));
  }

  void define(std::string_view name) {
    m_preamble.append(fmt::format("#define {} (1)\n", name));
  }

  CompilationResult compile();

private:
  GlslCompilerInstance(GlslCompiler *compiler,
                       std::unique_ptr<glslang::TShader> shader,
                       std::vector<std::byte> src, std::string sourcePath)
      : m_compiler(compiler), m_shader(std::move(shader)), m_preamble(),
        m_src(std::move(src)), m_sourcePath(std::move(sourcePath)) {}

private:
  GlslCompiler *m_compiler;
  std::unique_ptr<glslang::TShader> m_shader;
  std::string m_preamble;
  std::vector<std::byte> m_src;
  std::string m_sourcePath;
};

} // namespace denox::compiler
