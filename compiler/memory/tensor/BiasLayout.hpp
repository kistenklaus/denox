#pragma once

#include "diag/unreachable.hpp"
#include <cassert>
#include <cstddef>

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
    denox::compiler::diag::unreachable();
  }

  friend bool operator==(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_tag == rhs.m_tag;
  }
  friend bool operator!=(const BiasLayout &lhs, const BiasLayout &rhs) {
    return lhs.m_tag != rhs.m_tag;
  }

  BiasLayoutKind kind() const { return m_tag; }

private:
  BiasLayoutKind m_tag;
};
}; // namespace details::memory::tensors

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

private:
  details::memory::tensors::BiasLayout m_layout;
};

} // namespace denox::compiler
