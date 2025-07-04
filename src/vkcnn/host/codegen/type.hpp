#pragma once

#include <fmt/base.h>
namespace vkcnn::codegen {

enum class Type {
  F32,
  U32,
};

}

template <> struct fmt::formatter<vkcnn::codegen::Type> {
  constexpr auto parse(fmt::format_parse_context &ctx) const {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(vkcnn::codegen::Type t, FormatContext &ctx) const {
    const char *name = "";
    switch (t) {
    case vkcnn::codegen::Type::F32:
      name = "float";
      break;
    case vkcnn::codegen::Type::U32:
      name = "uint";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
