#pragma once

namespace vkcnn {

enum class SymExpr {
  // binary
  CeilDiv,  // (a + b - 1) / b
  FloorDiv, // a / b
  AlignUp,  // ((a+b-1)/b)*b   #b must be a power of 2!
  Mod,      // a % b
  Sub,      // a-b
  Mul,      // a*b*c
  Add,      // a+b+c
  Max,      // max(a,b,c)
  Min,      // min(a,b,c)
};

}
