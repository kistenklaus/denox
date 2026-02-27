#include "denox/io/fs/FileCache.hpp"
#include "denox/io/fs/File.hpp"
#include <mutex>

namespace denox::io {

memory::span<const std::byte> CachedFileHandle::bytes() const {
  std::lock_guard lck{m_fileCache->m_mutex};

  FileCache::CacheLine &line = m_fileCache->m_lines[m_cacheLine];
  // lazily load file.
  if (line.data == nullptr) {
    File file = File::open(line.path, File::OpenMode::Read);
    line.size = file.size();
    line.data = static_cast<std::byte *>(malloc(line.size));
    file.read_exact(memory::span<std::byte>{line.data, line.size});
  }
  return memory::span<const std::byte>{line.data, line.size};
}

} // namespace denox::io
