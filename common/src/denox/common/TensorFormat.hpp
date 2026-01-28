#pragma once

#include <fmt/core.h>

namespace denox {

enum class TensorFormat {
  Optimal,
  SSBO_HWC,
  SSBO_CHW,
  SSBO_CHWC8,
  TEX_RGBA,
  TEX_RGB,
  TEX_RG,
  TEX_R,
};

}

template <>
struct fmt::formatter<denox::TensorFormat> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin(); // no custom format specifiers
  }

  template <typename FormatContext>
  auto format(denox::TensorFormat fmt_type, FormatContext& ctx) const {
    using enum denox::TensorFormat;

    std::string_view name;
    switch (fmt_type) {
    case Optimal:     name = "Optimal";     break;
    case SSBO_HWC:    name = "HWC";    break;
    case SSBO_CHW:    name = "CHW";    break;
    case SSBO_CHWC8:  name = "CHWC8";  break;
    case TEX_RGBA:    name = "RGBA";    break;
    case TEX_RGB:     name = "RGB";     break;
    case TEX_RG:      name = "RG";      break;
    case TEX_R:       name = "R";       break;
    default:          name = "Unknown";     break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};
