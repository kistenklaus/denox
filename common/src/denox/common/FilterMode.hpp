#pragma once

#include <fmt/core.h>

namespace denox {

enum class FilterMode {
  Nearest,              // asymmetric
  Bilinear,             // half_pixel
  BilinearAlignCorners, // align_corners
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
    case denox::FilterMode::Bilinear:
      name = "bilinear";
      break;
    case denox::FilterMode::BilinearAlignCorners:
      name = "bilinear-align-corners";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
