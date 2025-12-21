#pragma once

#include <cstddef>
#include <fmt/core.h>

namespace denox::memory {

struct FilterShape {
  unsigned int s;
  unsigned int r;
  unsigned int c;
  unsigned int k;

  friend bool operator==(const FilterShape &lhs, const FilterShape &rhs) {
    return lhs.s == rhs.s && lhs.r == rhs.r && lhs.c == rhs.c && lhs.k == rhs.k;
  }
  friend bool operator!=(const FilterShape &lhs, const FilterShape &rhs) {
    return !(lhs == rhs);
  }

  std::size_t elemCount() const {
    return static_cast<std::size_t>(s) * static_cast<std::size_t>(r) *
           static_cast<std::size_t>(c) * static_cast<std::size_t>(k);
  }
};

} // namespace denox::memory

template <>
struct fmt::formatter<denox::memory::FilterShape> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::FilterShape& fs,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "{{s={}, r={}, c={}, k={}}}",
        fs.s, fs.r, fs.c, fs.k);
  }
};
