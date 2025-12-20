#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/memory/container/string_view.hpp"
#include "denox/spirv/CompilationResult.hpp"
#include <fmt/format.h>

namespace denox::spirv {

class GlslCompiler; // fwd declare

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
    std::size_t sourcePathHash = std::hash<const char *>{}(m_sourcePath.cstr());
    return algorithm::hash_combine(preambleHash, sourcePathHash);
  }

  std::string_view getPreamble() const { return m_preamble; }

  std::string key() const {
    return fmt::format("{}$&%;{}", m_sourcePath.str(), m_preamble);
  }

  uint64_t hashSrc() const {
    uint64_t hash = hashPreamble();
    for (std::byte b : m_src) {
      hash = algorithm::hash_combine(hash, std::hash<std::byte>{}(b));
    }
    return hash;
  }

  void sha256(SHA256Builder& hash) const {
    hash.update(std::span{reinterpret_cast<const uint8_t *>(m_preamble.data()),
                         m_preamble.size()});
    hash.update(std::span{reinterpret_cast<const uint8_t *>(m_src.data()),
                         m_src.size()});
  }

private:
  GlslCompilerInstance(GlslCompiler *compiler, memory::vector<std::byte> src,
                       io::Path sourcePath)
      : m_compiler(compiler), m_src(std::move(src)),
        m_sourcePath(std::move(sourcePath)) {}

private:
  GlslCompiler *m_compiler;
  memory::vector<std::byte> m_src;
  io::Path m_sourcePath;
  memory::string m_preamble;
  bool m_denoxPreprocessor = false;
};

} // namespace denox::spirv
