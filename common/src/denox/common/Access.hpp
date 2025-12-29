#pragma once

#include <fmt/format.h>

namespace denox {

enum class Access {
  ReadOnly = 1,
  WriteOnly = 2,
  ReadWrite = 3,
};

}

template <> struct fmt::formatter<denox::Access> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::Access access, FormatContext &ctx) const {
    const char *str = "unknown";
    switch (access) {
    case denox::Access::ReadOnly:
      str = "read_only";
      break;
    case denox::Access::WriteOnly:
      str = "write_only";
      break;
    case denox::Access::ReadWrite:
      str = "read_write";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", str);
  }
};
