#pragma once

#include <cstddef>
#include <cwchar>
#include <dnx.h>
#include <istream>

namespace denox::runtime {

struct Model {
  void* dnxBuffer;
  const dnx::Model *dnx;
};

} // namespace denox::runtime
