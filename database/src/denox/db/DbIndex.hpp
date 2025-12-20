#pragma once

#include "denox/common/SHA256.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/small_vector.hpp"

namespace denox {

struct DbIndex {
  memory::hash_map<SHA256, uint64_t> binary_index;
  memory::hash_map<uint64_t, memory::small_vector<uint64_t, 1>>
      dispatch_buckets;
};

} // namespace denox
