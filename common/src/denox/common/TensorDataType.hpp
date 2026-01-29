#pragma once
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/dtype/dtype.hpp"
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

static inline uint16_t align_of(TensorDataType dtype) {
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

static inline TensorDataType
tensor_data_type_from_memory_type(memory::Dtype type) {
  switch (type.kind()) {
  case memory::DtypeKind::F16:
    return TensorDataType::Float16;
  case memory::DtypeKind::F32:
    return TensorDataType::Float32;
  case memory::DtypeKind::F64:
    return TensorDataType::Float64;
  case memory::DtypeKind::U32:
  case memory::DtypeKind::I32:
  case memory::DtypeKind::U64:
  case memory::DtypeKind::I64:
    diag::invalid_state();
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
      denox::diag::unreachable();
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
