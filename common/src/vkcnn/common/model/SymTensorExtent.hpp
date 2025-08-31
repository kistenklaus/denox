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

  SymTensorExtent() : width(Sym::Const(0)), height(Sym::Const(0)) {}
  SymTensorExtent(Sym width, Sym height) : width(width), height(height) {}
};

} // namespace vkcnn
