#pragma once

#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/string.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>

namespace denox::memory {

enum class DtypeKind : std::uint8_t {
  F16 = 0,
  F32 = 1,
  F64 = 2,
  U32 = 3,
  I32 = 4,
  U64 = 5,
  I64 = 6
};

namespace details::dtype {

struct Dtype {
  static constexpr std::array<std::size_t, 7> sizes = {2, 4, 8, 4, 4, 8, 8};
  static constexpr std::array<std::size_t, 7> alignments = {2, 4, 8, 4,
                                                            4, 8, 8};

  constexpr Dtype(DtypeKind kind) : m_kind(kind) {}

  std::size_t size() const {
    return sizes[static_cast<std::underlying_type_t<DtypeKind>>(m_kind)];
  }

  std::size_t alignment() const {
    return alignments[static_cast<std::underlying_type_t<DtypeKind>>(m_kind)];
  }

  friend bool operator==(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_kind == rhs.m_kind;
  }
  friend bool operator!=(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_kind != rhs.m_kind;
  }

  DtypeKind kind() const { return m_kind; }

private:
  DtypeKind m_kind;
};

} // namespace details::dtype

class Dtype {
public:
  constexpr Dtype(details::dtype::Dtype type) : m_type(type) {}
  constexpr Dtype() : m_type(DtypeKind::F16) {}

  static constexpr details::dtype::Dtype F16{DtypeKind::F16};
  static constexpr details::dtype::Dtype F32{DtypeKind::F32};
  static constexpr details::dtype::Dtype F64{DtypeKind::F64};
  static constexpr details::dtype::Dtype U32{DtypeKind::U32};
  static constexpr details::dtype::Dtype I32{DtypeKind::I32};

  static constexpr details::dtype::Dtype U64{DtypeKind::U64};
  static constexpr details::dtype::Dtype I64{DtypeKind::I64};

  std::size_t size() const { return m_type.size(); }
  std::size_t alignment() const { return m_type.alignment(); }

  friend bool operator==(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_type == rhs.m_type;
  }

  friend bool operator!=(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_type != rhs.m_type;
  }

  memory::string to_string() const {
    switch (m_type.kind()) {
    case DtypeKind::F16:
      return "float16";
    case DtypeKind::F32:
      return "float32";
    case DtypeKind::F64:
      return "float64";
    case DtypeKind::U32:
      return "uint32";
    case DtypeKind::I32:
      return "uint32";
    case DtypeKind::U64:
      return "uint64";
    case DtypeKind::I64:
      return "int64";
    default:
      diag::unreachable();
    }
  }

  DtypeKind kind() const { return m_type.kind(); }

private:
  details::dtype::Dtype m_type;
};

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::details::dtype::Dtype> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::details::dtype::Dtype &dt,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{}",
                          denox::memory::Dtype{dt}.to_string());
  }
};

template <> struct fmt::formatter<denox::memory::Dtype> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::Dtype &dt, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{}", dt.to_string());
  }
};
