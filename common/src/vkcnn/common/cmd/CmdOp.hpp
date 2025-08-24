#pragma once

#include <cstdint>
namespace vkcnn {

enum class CmdOp : std::uint8_t{
  NoOp,
  Dispatch,
  Barrier,
  Last = Barrier,
};

} // namespace vkcnn
