#pragma once

#include "denox/diag/unreachable.hpp"
#include <cassert>
#include <cstddef>
#include <fmt/core.h>

namespace denox::memory {

enum class BiasLayoutKind {
  C,
  C4,
  C8,
  C16,
};

namespace details::memory::tensors {

class BiasLayout {
public:
  constexpr BiasLayout(BiasLayoutKind tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()([[maybe_unused]] unsigned int shape, unsigned int c) const {
    assert(c < shape);
    return c;
  }

  __attribute__((always_inline)) inline constexpr std::size_t
  size(unsigned int shape) const {
    const std::size_t N = static_cast<std::size_t>(shape);
    switch (m_tag) {
    case BiasLayoutKind::C:
      return N;

    case BiasLayoutKind::C4: // round up to multiple of 4
      return (N + std::size_t{3}) & ~std::size_t{3};

    case BiasLayoutKind::C8: // round up to multiple of 8
      return (N + std::size_t{7}) & ~std::size_t{7};

    case BiasLayoutKind::C16: // round up to multiple of 16
      return (N + std::size_t{15}) & ~std::size_t{15};
    }
    diag::unreachable();
  }

  friend bool operator==(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_tag == rhs.m_tag;
  }
  friend bool operator!=(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_tag != rhs.m_tag;
  }

  BiasLayoutKind kind() const { return m_tag; }

  bool isVectorized() const {
    switch (m_tag) {
    case BiasLayoutKind::C:
      return false;
    case BiasLayoutKind::C4:
    case BiasLayoutKind::C8:
    case BiasLayoutKind::C16:
      return true;
    default:
      diag::unreachable();
    }
  }

private:
  BiasLayoutKind m_tag;
};

} // namespace details::memory::tensors

class BiasLayout {
public:
  constexpr BiasLayout(details::memory::tensors::BiasLayout layout)
      : m_layout(layout) {}

  static constexpr details::memory::tensors::BiasLayout C{BiasLayoutKind::C};
  static constexpr details::memory::tensors::BiasLayout C4{BiasLayoutKind::C4};
  static constexpr details::memory::tensors::BiasLayout C8{BiasLayoutKind::C8};
  static constexpr details::memory::tensors::BiasLayout C16{
      BiasLayoutKind::C16};

  constexpr std::size_t operator()(unsigned int shape, unsigned int c) const {
    return m_layout(shape, c);
  }

  constexpr std::size_t size(unsigned int shape) const {
    return m_layout.size(shape);
  }

  friend bool operator==(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_layout == rhs.m_layout;
  }

  friend bool operator!=(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_layout == rhs.m_layout;
  }

  BiasLayoutKind kind() const { return m_layout.kind(); }

  bool isVectorized() const { return m_layout.isVectorized(); }

private:
  details::memory::tensors::BiasLayout m_layout;
};

} // namespace denox::memory

template <>
struct fmt::formatter<denox::memory::details::memory::tensors::BiasLayout> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::details::memory::tensors::BiasLayout &layout,
              FormatContext &ctx) const {

    const char *name = nullptr;
    switch (layout.kind()) {
    case denox::memory::BiasLayoutKind::C:
      name = "C";
      break;
    case denox::memory::BiasLayoutKind::C4:
      name = "C4";
      break;
    case denox::memory::BiasLayoutKind::C8:
      name = "C8";
      break;
    case denox::memory::BiasLayoutKind::C16:
      name = "C16";
      break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};

template <>
struct fmt::formatter<denox::memory::BiasLayout> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::BiasLayout& layout,
              FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", layout.kind());
  }
};

template <>
struct fmt::formatter<denox::memory::BiasLayoutKind> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(denox::memory::BiasLayoutKind kind,
              FormatContext& ctx) const {
    switch (kind) {
    case denox::memory::BiasLayoutKind::C:
      return fmt::format_to(ctx.out(), "C");
    case denox::memory::BiasLayoutKind::C4:
      return fmt::format_to(ctx.out(), "C4");
    case denox::memory::BiasLayoutKind::C8:
      return fmt::format_to(ctx.out(), "C8");
    case denox::memory::BiasLayoutKind::C16:
      return fmt::format_to(ctx.out(), "C16");
    }
    // unreachable, but keeps compilers happy
    return ctx.out();
  }
};
