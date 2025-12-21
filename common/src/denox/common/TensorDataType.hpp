#pragma once
#include <fmt/core.h>

namespace denox {

enum class TensorDataType {
  Auto,
  Float16,
  Float32,
  Float64,
};

}

template <>
struct fmt::formatter<denox::TensorDataType> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin(); // no custom format specifiers
  }

  template <typename FormatContext>
  auto format(denox::TensorDataType type, FormatContext& ctx) const {
    using enum denox::TensorDataType;

    std::string_view name;
    switch (type) {
    case Auto:    name = "Auto";    break;
    case Float16: name = "Float16"; break;
    case Float32: name = "Float32"; break;
    case Float64: name = "Float64"; break;
    default:      name = "Unknown"; break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};
