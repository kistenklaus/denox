#pragma once

#include <cstdint>
namespace vkcnn {

struct CmdRecordBarrier {
  unsigned int inputTensor;
  unsigned int outputTensor;
};

static_assert(alignof(std::uint64_t) >= alignof(CmdRecordBarrier));

} // namespace vkcnn
