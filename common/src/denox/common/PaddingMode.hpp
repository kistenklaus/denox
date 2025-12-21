#pragma once

#include <fmt/core.h>

namespace denox {

enum class PaddingMode {
  Zero,
  Edge,
};

}

template <> struct fmt::formatter<denox::PaddingMode> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::PaddingMode mode, FormatContext &ctx) const {
    const char *name = nullptr;
    switch (mode) {
    case denox::PaddingMode::Zero:
      name = "zero";
      break;
    case denox::PaddingMode::Edge:
      name = "edge";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
