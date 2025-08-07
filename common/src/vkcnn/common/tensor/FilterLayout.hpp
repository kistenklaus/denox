#pragma once

#include "vkcnn/common/tensor/FilterShape.hpp"
#include <cassert>
#include <compare>
#include <cstddef>
#include <stdexcept>
namespace vkcnn {

namespace details {

class FilterLayout {
public:
  enum class Tag {
    KRSC,
    KCRS,
    RSCK,
    RSKC,
    RSCKC8,
    RCSKC8,
    RSCKC16,
    RCSKC16,
    RSKCK8,
    RSKCK16,
    KRSCK8,
    KRSCK16,
  };
  constexpr FilterLayout(Tag tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(const FilterShape &shape, unsigned int s, unsigned r,
             unsigned int c, unsigned int k) const {
    // TODO do all math in size_t instead of unsigned integers.
    switch (m_tag) {
    case Tag::KCRS:
      return k * (shape.c * shape.r * shape.s) + c * (shape.r * shape.s) +
             r * (shape.s) + s;
    case Tag::KRSC:
      return k * (shape.r * shape.s * shape.c) + r * (shape.s * shape.c) +
             s * (shape.c) + c;
    case Tag::RSCK:
      return r * (shape.s * shape.c * shape.k) + s * (shape.c * shape.k) +
             c * (shape.k) + k;
    case Tag::RSKC:
      return r * (shape.s * shape.c * shape.k) + s * (shape.c * shape.k) +
             k * (shape.c) + c;
    case Tag::RSCKC8:
      assert(shape.c % 8 == 0);
      return r * (shape.s * shape.c * shape.k) + s * (shape.c * shape.k) +
             (c / 8) * (shape.k * 8) + k * (8) + (c % 8);
    case Tag::RCSKC8:
      assert(shape.c % 8 == 0);
      return r * (shape.c * shape.s * shape.k) +
             (c / 8) * (shape.s * shape.k * 8) + s * (shape.k * 8) + k * 8 +
             (c % 8);
    case Tag::RSCKC16:
      assert(shape.c % 16 == 0);
      return r * (shape.s * shape.c * shape.k) + s * (shape.c * shape.k) +
             (c / 16) * (shape.k * 16) + k * (16) + (c % 16);
    case Tag::RCSKC16:
      assert(shape.c % 16 == 0);
      return r * (shape.c * shape.s * shape.k) +
             (c >> 4) * (shape.s * shape.k << 4) + s * (shape.k << 4) +
             (k << 4) + (c & 0xF);
    case Tag::RSKCK8:
      assert(shape.k % 8 == 0);
      return r * (shape.c * shape.s * shape.k) + s * (shape.c * shape.k) +
             (k / 8) * (shape.c * 8) + c * 8 + k % 8;
    case Tag::RSKCK16:
      assert(shape.k % 16 == 0);
      return r * (shape.c * shape.s * shape.k) + s * (shape.c * shape.k) +
             (k / 16) * (shape.c * 16) + c * 16 + k % 16;
    case Tag::KRSCK8:
      assert(shape.k % 8 == 0);
      return (k / 8) * (shape.r * shape.s * shape.c * 8) +
             r * (shape.s * shape.c * 8) + s * (shape.c * 8) + c * 8 + (k % 8);
    case Tag::KRSCK16:
      assert(shape.k % 16 == 0);
      return (k / 16) * (shape.r * shape.s * shape.c * 16) +
             r * (shape.s * shape.c * 16) + s * (shape.c * 16) + c * 16 +
             (k % 16);
    }
    throw std::runtime_error("Unsupported FilterLayout");
  }

  constexpr auto operator<=>(const FilterLayout &) const = default;

private:
  Tag m_tag;
};

}; // namespace details

class FilterLayout {
public:
  constexpr FilterLayout(details::FilterLayout layout) : m_layout(layout) {}

  static constexpr details::FilterLayout KRSC{details::FilterLayout::Tag::KRSC};

  static constexpr details::FilterLayout KCRS{details::FilterLayout::Tag::KCRS};

  static constexpr details::FilterLayout RSCK{details::FilterLayout::Tag::RSCK};

  static constexpr details::FilterLayout RSKC{details::FilterLayout::Tag::RSKC};

  static constexpr details::FilterLayout RSCKC8{
      details::FilterLayout::Tag::RSCKC8};

  static constexpr details::FilterLayout RCSKC8{
      details::FilterLayout::Tag::RCSKC8};

  static constexpr details::FilterLayout RSCKC16{
      details::FilterLayout::Tag::RSCKC16};

  static constexpr details::FilterLayout RCSKC16{
      details::FilterLayout::Tag::RCSKC16};

  static constexpr details::FilterLayout RSKCK8{
      details::FilterLayout::Tag::RSKCK8};
  static constexpr details::FilterLayout RSKCK16{
      details::FilterLayout::Tag::RSKCK16};

  static constexpr details::FilterLayout KRSCK8{
      details::FilterLayout::Tag::KRSCK8};
  static constexpr details::FilterLayout KRSCK16{
      details::FilterLayout::Tag::KRSCK16};

  constexpr std::size_t operator()(const FilterShape &shape, unsigned int s,
                                   unsigned r, unsigned int c,
                                   unsigned int k) const {
    return m_layout(shape, s, r, c, k);
  }

  constexpr auto operator<=>(const FilterLayout &) const = default;

private:
  details::FilterLayout m_layout;
};
} // namespace vkcnn
