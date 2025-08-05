#pragma once

#include <cassert>
#include <compare>
#include <cstddef>
#include <stdexcept>
#include <utility>
namespace vkcnn {

namespace details {

class BiasLayout {
public:
  enum class Tag {
    C,
    C4,
    C8,
    C16,
  };

  constexpr BiasLayout(Tag tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(unsigned int shape, unsigned int c) const {
    assert(c < shape);
    return c;
  }

  __attribute__((always_inline)) inline constexpr std::size_t
  size(unsigned int shape) const {
    // NOTE: unsigned under/ overflow is defined behavior!!!.
    // This does not work with signed integers.
    switch (m_tag) {
    case Tag::C:
      return shape;
    case Tag::C4:
      return ((shape + 4 - 1) / 4) * 4; // pad to a multiple of 4.
    case Tag::C8:
      return ((shape + 8 - 1) / 8) * 8; // pad to a multiple of 4.
    case Tag::C16:
      return ((shape + 16 - 1) / 16) * 16; // pad to a multiple of 4.
    default:
      std::unreachable();
      throw std::runtime_error("NOT-IMPLEMENTED");
    }
  }

  constexpr auto operator<=>(const BiasLayout &) const = default;

private:
  Tag m_tag;
};
}; // namespace details

class BiasLayout {
public:
  constexpr BiasLayout(details::BiasLayout layout) : m_layout(layout) {}

  static constexpr details::BiasLayout C{details::BiasLayout::Tag::C};
  static constexpr details::BiasLayout C4{details::BiasLayout::Tag::C4};
  static constexpr details::BiasLayout C8{details::BiasLayout::Tag::C8};
  static constexpr details::BiasLayout C16{details::BiasLayout::Tag::C16};

  constexpr std::size_t operator()(unsigned int shape, unsigned int c) const {
    return m_layout(shape, c);
  }

  constexpr std::size_t size(unsigned int shape) const {
    return m_layout.size(shape);
  }

  constexpr auto operator<=>(const BiasLayout &) const = default;

private:
  details::BiasLayout m_layout;
};

} // namespace vkcnn
