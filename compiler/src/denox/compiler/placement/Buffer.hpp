#pragma once

#include "denox/symbolic/Sym.hpp"
namespace denox::compiler {

struct Buffer {
  Sym size;
  uint16_t alignment;

  // beginning of lifetime (indexes dispatches)
  uint32_t first_use;

  // end of lifetime (indexes dispatches)
  uint32_t last_use;
};

} // namespace denox::compiler
