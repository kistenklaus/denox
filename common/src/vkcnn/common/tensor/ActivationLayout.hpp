#pragma once

#include "vkcnn/common/tensor/ActivationShape.hpp"
#include <compare>
#include <cstddef>

namespace vkcnn {

// NOTE: A weirdly complex enum class wrapper, only that we can use operator()
// cleanly.

namespace details {
class ActivationLayout {
public:
  enum class Tag { CHW, HWC, CHWC8 };
  constexpr ActivationLayout(Tag tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(const ActivationShape &shape, unsigned int w, unsigned h,
             unsigned int c) const {
    // TODO do all math in size_t instead of unsigned integers.
    switch (m_tag) {
    case Tag::HWC:
      return h * (shape.w * shape.c) + w * (shape.c) + c;
    case Tag::CHW:
      return c * (shape.h * shape.w) + h * (shape.w) + w;
    case Tag::CHWC8:
      return (c >> 3) * (shape.h * shape.w * 8) + h * (shape.w * 8) + w * 8 +
             (c & 0x7);
    }
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

  static constexpr details::ActivationLayout CHWC8{
      details::ActivationLayout::Tag::CHWC8};

  constexpr std::size_t operator()(const ActivationShape &shape, unsigned int w,
                                   unsigned h, unsigned int c) const {
    return m_layout(shape, w, h, c);
  }

  constexpr auto operator<=>(const ActivationLayout &) const = default;

private:
  details::ActivationLayout m_layout;
};

} // namespace vkcnn
