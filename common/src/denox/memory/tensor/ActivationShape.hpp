#pragma once

#include <fmt/core.h>

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

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::ActivationShape> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::ActivationShape &s,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{w={}, h={}, c={}}}", s.w, s.h, s.c);
  }
};
