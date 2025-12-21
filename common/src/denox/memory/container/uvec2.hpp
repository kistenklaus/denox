#pragma once

#include <fmt/core.h>

namespace denox::memory {

struct uvec2 {
  unsigned int x;
  unsigned int y;

  friend bool operator==(const uvec2& lhs, const uvec2& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }
  friend bool operator!=(const uvec2& lhs, const uvec2& rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::memory

template <>
struct fmt::formatter<denox::memory::uvec2> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::uvec2& v, FormatContext& ctx) const {
    return fmt::format_to(
      ctx.out(),
      "{{x={}, y={}}}",
      v.x,
      v.y
    );
  }
};
