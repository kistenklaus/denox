#pragma once
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include <fmt/core.h>

namespace denox {

enum class TensorDataType {
  Auto,
  Float16,
  Float32,
  Float64,
};

static inline size_t size_of(TensorDataType dtype) {
  switch (dtype) {
  case TensorDataType::Auto:
    diag::invalid_state();
  case TensorDataType::Float16:
    return 2;
  case TensorDataType::Float32:
    return 4;
  case TensorDataType::Float64:
    return 8;
  }
  diag::unreachable();
}

static inline size_t align_of(TensorDataType dtype) {
  switch (dtype) {
  case TensorDataType::Auto:
    diag::invalid_state();
  case TensorDataType::Float16:
    return 2;
  case TensorDataType::Float32:
    return 4;
  case TensorDataType::Float64:
    return 8;
  }
  diag::unreachable();
}
} // namespace denox

template <> struct fmt::formatter<denox::TensorDataType> {
  constexpr auto parse(fmt::format_parse_context &ctx) {
    return ctx.begin(); // no custom format specifiers
  }

  template <typename FormatContext>
  auto format(denox::TensorDataType type, FormatContext &ctx) const {
    using enum denox::TensorDataType;

    std::string_view name;
    switch (type) {
    case Auto:
      name = "Auto";
      break;
    case Float16:
      name = "Float16";
      break;
    case Float32:
      name = "Float32";
      break;
    case Float64:
      name = "Float64";
      break;
    default:
      name = "Unknown";
      break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};
