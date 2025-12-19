#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/memory/container/string_view.hpp"
#include "shaders/compiler/CompilationResult.hpp"
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
    requires(fmt::is_formattable<T>::value &&
             !std::same_as<T, memory::string_view> &&
             !std::same_as<T, memory::string> &&
             !std::same_as<T, const char *> && !std::same_as<T, bool>)
  void define(std::string_view name, const T &value) {
    m_preamble.append(fmt::format("#define {} ({})\n", name, value));
  }

  void define(std::string_view name, memory::string_view value) {
    m_preamble.append(fmt::format("#define {} {}\n", name, value));
  }

  void define(std::string_view name, memory::string value) {
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

  void enableDenoxPreprocessor() { m_denoxPreprocessor = true; }

  CompilationResult compile();

  const io::Path &getSourcePath() const { return m_sourcePath; }

  std::size_t hashPreamble() const {
    std::size_t preambleHash = std::hash<memory::string>{}(m_preamble);
    std::size_t sourcePathHash =
        std::hash<memory::string>{}(m_sourcePath.str());
    return algorithm::hash_combine(preambleHash, sourcePathHash);
  }

  std::string_view getPreamble() const { return m_preamble; }

  std::string key() const {
    return fmt::format("{}$&%;{}", m_sourcePath.str(), m_preamble);
  }

private:
  GlslCompilerInstance(GlslCompiler *compiler,
                       std::unique_ptr<glslang::TShader> shader,
                       memory::vector<std::byte> src, io::Path sourcePath)
      : m_compiler(compiler), m_shader(std::move(shader)), m_preamble(),
        m_denoxPreprocessor(true), m_src(std::move(src)),
        m_sourcePath(std::move(sourcePath)) {}

private:
  GlslCompiler *m_compiler;
  std::unique_ptr<glslang::TShader> m_shader;
  memory::string m_preamble;
  bool m_denoxPreprocessor;
  memory::vector<std::byte> m_src;
  io::Path m_sourcePath;
};

} // namespace denox::compiler
