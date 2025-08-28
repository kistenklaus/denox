#pragma once

#include "vkcnn/common/symbolic/Sym.hpp"

namespace vkcnn {

struct SymTensorExtent {
  Sym width;
  Sym height;

  friend bool operator==(const SymTensorExtent &,
                         const SymTensorExtent &) = default;
  friend bool operator!=(const SymTensorExtent &,
                         const SymTensorExtent &) = default;
};

} // namespace vkcnn
