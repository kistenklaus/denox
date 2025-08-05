#pragma once

#include "vkcnn/common/tensor/ActivationShape.hpp"
#include <compare>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace vkcnn {

// NOTE: A weirdly complex enum class wrapper, only that we can use operator()
// cleanly.

namespace details {
class ActivationLayout {
public:
  enum class Tag { CHW, HWC, CHWC4, CHWC8, CHWC16 };
  constexpr ActivationLayout(Tag tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(const ActivationShape &shape, unsigned int w, unsigned h,
             unsigned int c) const {
    // TODO do all math in int64_t instead of unsigned integers.
    // for better arithmetic optimizations and to avoid overflows.
    switch (m_tag) {
    case Tag::HWC:
      return h * (shape.w * shape.c) + w * (shape.c) + c;
    case Tag::CHW:
      return c * (shape.h * shape.w) + h * (shape.w) + w;
    case Tag::CHWC4:
      return (c >> 2) * (shape.h * shape.w << 2) + h * (shape.w << 2) +
             (w << 2) + (c & 0x3);
    case Tag::CHWC8:
      return (c >> 3) * (shape.h * shape.w << 3) + h * (shape.w << 3) +
             (w << 3) + (c & 0x7);
    case Tag::CHWC16:
      return (c >> 4) * (shape.h * shape.w << 4) + h * (shape.w << 4) +
             (w << 4) + (c & 0xF);
    }
    throw std::runtime_error("Invalid layout");
  }

  constexpr auto operator<=>(const ActivationLayout &) const = default;

private:
  Tag m_tag;
};
}; // namespace details

class ActivationLayout {
public:
  constexpr ActivationLayout(details::ActivationLayout layout)
      : m_layout(layout) {}
  static constexpr details::ActivationLayout HWC{
      details::ActivationLayout::Tag::HWC};
  static constexpr details::ActivationLayout CHW{
      details::ActivationLayout::Tag::CHW};

  static constexpr details::ActivationLayout CHWC4{
      details::ActivationLayout::Tag::CHWC4};

  static constexpr details::ActivationLayout CHWC8{
      details::ActivationLayout::Tag::CHWC8};

  static constexpr details::ActivationLayout CHWC16{
      details::ActivationLayout::Tag::CHWC16};

  constexpr std::size_t operator()(const ActivationShape &shape, unsigned int w,
                                   unsigned h, unsigned int c) const {
    return m_layout(shape, w, h, c);
  }

  constexpr auto operator<=>(const ActivationLayout &) const = default;

private:
  details::ActivationLayout m_layout;
};

} // namespace vkcnn
