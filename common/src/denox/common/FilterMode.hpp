#pragma once

#include <fmt/core.h>

namespace denox {

enum class FilterMode {
  Nearest,
};

}

template <> struct fmt::formatter<denox::FilterMode> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::FilterMode mode, FormatContext &ctx) const {
    const char *name = nullptr;
    switch (mode) {
    case denox::FilterMode::Nearest:
      name = "nearest";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
