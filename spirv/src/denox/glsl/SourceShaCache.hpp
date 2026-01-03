#pragma once

#include "denox/common/SHA256.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/hashmap.hpp"
namespace denox::spirv {

class SourceShaCache {
public:
  SHA256Builder cachedSHA(const io::Path &path,
                          std::span<const std::byte> source) {
    if (m_cache.contains(path)) {
      return m_cache.at(path);
    } else {
      SHA256Builder hasher;
      hasher.update(std::span{reinterpret_cast<const uint8_t *>(source.data()),
                              source.size()});
      m_cache[path] = hasher;
      return hasher;
    }
  }

private:
  memory::hash_map<io::Path, SHA256Builder> m_cache;
};

} // namespace denox::spirv
