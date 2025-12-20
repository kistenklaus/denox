#pragma once

namespace denox::memory {

struct ActivationShape {
  unsigned int w;
  unsigned int h;
  unsigned int c;

  friend bool operator==(const ActivationShape &lhs,
                         const ActivationShape &rhs) {
    return lhs.w == rhs.w && lhs.h == rhs.h && lhs.c == rhs.c;
  }

  friend bool operator!=(const ActivationShape &lhs,
                         const ActivationShape &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::compiler
