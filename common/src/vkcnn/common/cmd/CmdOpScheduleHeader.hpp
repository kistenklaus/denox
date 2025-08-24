#pragma once

#include <cstddef>
namespace vkcnn {

struct CmdOpScheduleHeader {
  std::size_t op_offset;
  std::size_t op_count;
  std::size_t cmdOT_offset;
  std::size_t cmdBuf_offset;
  std::size_t spirv_offset;
  std::size_t param_offset;
  std::size_t byteSize;
};

} // namespace vkcnn
