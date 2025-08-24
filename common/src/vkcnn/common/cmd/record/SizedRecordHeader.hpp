#pragma once

#include <cstdint>
namespace vkcnn {

struct SizedRecordHeader {
  std::uint64_t size; // <- in 4 byte words for spirv and 1 byte for weights
};

} // namespace vkcnn
