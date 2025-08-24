#pragma once

#include <cstdint>
namespace vkcnn {

struct CmdRecordDispatch {
  unsigned int inputTensor;
  unsigned int outputTensor;
  std::uint64_t spirvOffset;
  unsigned int parameterCount;
};
static_assert(alignof(std::uint64_t) >= alignof(CmdRecordDispatch));

}; // namespace vkcnn
