#pragma once

#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include <algorithm>
#include <mutex>

namespace denox::io {

class FileCache;

class CachedFileHandle {
public:
  friend class FileCache;
  memory::span<const std::byte> bytes() const;

private:
  CachedFileHandle(size_t cacheLine, FileCache *cache)
      : m_cacheLine(cacheLine), m_fileCache(cache) {}
  size_t m_cacheLine;
  FileCache *m_fileCache;
};

class FileCache {
public:
  friend class CachedFileHandle;

  FileCache() : m_mutex() {}

  ~FileCache() { reset(); }

  // File cache is not movable because pointer of it have to be stable.
  FileCache(const FileCache &) = delete;
  FileCache &operator=(const FileCache &) = delete;
  FileCache(FileCache &&) = delete;
  FileCache &operator=(FileCache &&) = delete;

  CachedFileHandle read(const Path &path) {
    std::lock_guard lck{m_mutex};
    auto it =
        std::ranges::find_if(m_lines, [&path](const CacheLine &line) -> bool {
          return line.path == path;
        });
    if (it == m_lines.end()) {
      m_lines.push_back(CacheLine{
          .path = path,
          .data = nullptr,
          .size = 0,
      });
      return CachedFileHandle(m_lines.size() - 1, this);
    } else {
      return CachedFileHandle(
          static_cast<size_t>(std::distance(m_lines.begin(), it)), this);
    }
  }

  // NOTE: Invalidates all FileHandles
  void reset() {
    for (const CacheLine &line : m_lines) {
      free(line.data);
    }
  }

private:
  struct CacheLine {
    Path path;
    std::byte *data;
    size_t size;
  };

  memory::vector<CacheLine> m_lines;
  std::mutex m_mutex;
};

} // namespace denox::io
