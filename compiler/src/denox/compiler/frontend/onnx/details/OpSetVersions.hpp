#pragma once

#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/string.hpp"
#include <cassert>
#include <cstdint>

namespace denox::onnx {

using opset_version = std::int64_t;
struct OpSetVersions {
  opset_version core_version;

  opset_version operator[](const denox::memory::string &domain) {
    assert(map.contains(domain));
    return map[domain];
  }

  denox::memory::hash_map<std::string, opset_version> map;
};

} // namespace denox::onnx
