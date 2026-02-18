#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/io/fs/FileCache.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/string.hpp"
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
    for (std::byte b : m_srcHandle.bytes()) {
      hash = algorithm::hash_combine(hash, std::hash<std::byte>{}(b));
    }
    return hash;
  }

  void sha256(SHA256Builder &hash) const {
    memory::span<const std::byte> src = m_srcHandle.bytes();
    hash.update(std::span{reinterpret_cast<const uint8_t *>(src.data()),
                          src.size()});

    hash.update(std::span{reinterpret_cast<const uint8_t *>(m_preamble.data()),
                          m_preamble.size()});
  }

  SHA256 fast_sha256() const;

private:
  GlslCompilerInstance(GlslCompiler *compiler, io::CachedFileHandle src,
                       io::Path sourcePath)
      : m_compiler(compiler), m_srcHandle(std::move(src)),
        m_sourcePath(std::move(sourcePath)),
        m_mutex(std::make_shared<std::mutex>()) {}

private:
  GlslCompiler *m_compiler;
  io::CachedFileHandle m_srcHandle;
  io::Path m_sourcePath;
  memory::string m_preamble;
  bool m_denoxPreprocessor = true;
  std::shared_ptr<std::mutex> m_mutex;
};

} // namespace denox::spirv
