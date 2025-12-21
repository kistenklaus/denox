#pragma once

#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/tensor/ActivationShape.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>

namespace denox::memory {

enum class ActivationLayoutKind : uint64_t {
  CHW,
  HWC,
  HWC8,
  CHWC4,
  CHWC8,
  CHWC16
};

namespace details::memory::tensors {
class ActivationLayout {
public:
  constexpr ActivationLayout(ActivationLayoutKind tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(const ActivationShape &shape, unsigned int w, unsigned int h,
             unsigned int c) const {
    const std::size_t SW = static_cast<std::size_t>(shape.w);
    const std::size_t SH = static_cast<std::size_t>(shape.h);
    const std::size_t SC = static_cast<std::size_t>(shape.c);
    const std::size_t W = static_cast<std::size_t>(w);
    const std::size_t H = static_cast<std::size_t>(h);
    const std::size_t C = static_cast<std::size_t>(c);

    switch (m_tag) {
    case ActivationLayoutKind::HWC8:
    case ActivationLayoutKind::HWC:
      return H * (SW * SC) + W * SC + C;

    case ActivationLayoutKind::CHW:
      return C * (SH * SW) + H * SW + W;

    case ActivationLayoutKind::CHWC4: {
      assert(SC % 4u == 0u);
      const std::size_t HW = SH * SW;
      return (C >> 2) * (HW << 2) + H * (SW << 2) + (W << 2) +
             (C & std::size_t{0x3});
    }

    case ActivationLayoutKind::CHWC8: {
      assert(SC % 8u == 0u);
      const std::size_t HW = SH * SW;
      return (C >> 3) * (HW << 3) + H * (SW << 3) + (W << 3) +
             (C & std::size_t{0x7});
    }

    case ActivationLayoutKind::CHWC16: {
      assert(SC % 16u == 0u);
      const std::size_t HW = SH * SW;
      return (C >> 4) * (HW << 4) + H * (SW << 4) + (W << 4) +
             (C & std::size_t{0xF});
    }
    }
    denox::diag::unreachable();
  }

  bool supports(unsigned int channels) const {
    switch (m_tag) {
    case ActivationLayoutKind::CHW:
      return true;
    case ActivationLayoutKind::HWC:
      return true;
    case ActivationLayoutKind::HWC8:
      return channels % 8 == 0;
    case ActivationLayoutKind::CHWC4:
      return channels % 4 == 0;
    case ActivationLayoutKind::CHWC8:
      return channels % 8 == 0;
    case ActivationLayoutKind::CHWC16:
      return channels % 16 == 0;
    default:
      diag::unreachable();
    }
  }

  std::string to_string() const {
    switch (m_tag) {
    case ActivationLayoutKind::CHW:
      return "CHW";
    case ActivationLayoutKind::HWC:
      return "HWC";
    case ActivationLayoutKind::HWC8:
      return "HWC8";
    case ActivationLayoutKind::CHWC4:
      return "CHWC4";
    case ActivationLayoutKind::CHWC8:
      return "CHWC8";
    case ActivationLayoutKind::CHWC16:
      return "CHWC16";
    default:
      diag::unreachable();
    }
  }

  bool isVectorized() const {
    switch (m_tag) {
    case ActivationLayoutKind::CHW:
    case ActivationLayoutKind::HWC:
      return false;
    case ActivationLayoutKind::HWC8:
    case ActivationLayoutKind::CHWC4:
    case ActivationLayoutKind::CHWC8:
    case ActivationLayoutKind::CHWC16:
      return true;
    default:
      diag::unreachable();
    }
  }

  unsigned int vectorBlockSize() const {
    switch (m_tag) {
    case ActivationLayoutKind::CHW:
    case ActivationLayoutKind::HWC:
      return 1;
    case ActivationLayoutKind::CHWC4:
      return 4;
    case ActivationLayoutKind::HWC8:
    case ActivationLayoutKind::CHWC8:
      return 8;
    case ActivationLayoutKind::CHWC16:
      return 16;
    default:
      diag::unreachable();
    }
  }

  friend bool operator==(const ActivationLayout &lhs,
                         const ActivationLayout &rhs) {
    return lhs.m_tag == rhs.m_tag;
  }

  friend bool operator!=(const ActivationLayout &lhs,
                         const ActivationLayout &rhs) {
    return lhs.m_tag != rhs.m_tag;
  }

  ActivationLayoutKind kind() const { return m_tag; }

private:
  ActivationLayoutKind m_tag;
};

} // namespace details::memory::tensors

class ActivationLayout {
public:
  constexpr ActivationLayout(details::memory::tensors::ActivationLayout layout)
      : m_layout(layout) {}
  static constexpr details::memory::tensors::ActivationLayout HWC{
      ActivationLayoutKind::HWC};

  static constexpr details::memory::tensors::ActivationLayout CHW{
      ActivationLayoutKind::CHW};

  static constexpr details::memory::tensors::ActivationLayout CHWC4{
      ActivationLayoutKind::CHWC4};

  static constexpr details::memory::tensors::ActivationLayout CHWC8{
      ActivationLayoutKind::CHWC8};

  static constexpr details::memory::tensors::ActivationLayout CHWC16{
      ActivationLayoutKind::CHWC16};

  constexpr std::size_t operator()(const ActivationShape &shape, unsigned int w,
                                   unsigned h, unsigned int c) const {
    return m_layout(shape, w, h, c);
  }

  friend bool operator==(const ActivationLayout &lhs,
                         const ActivationLayout &rhs) {
    return lhs.m_layout == rhs.m_layout;
  }

  friend bool operator!=(const ActivationLayout &lhs,
                         const ActivationLayout &rhs) {
    return lhs.m_layout != rhs.m_layout;
  }

  ActivationLayoutKind kind() const { return m_layout.kind(); }

  static memory::span<const ActivationLayout> supported() {
    static constexpr ActivationLayout layouts[] = {HWC, CHWC8};
    return layouts;
  }

  std::string to_string() const { return m_layout.to_string(); }

  bool supports(unsigned int channels) const {
    return m_layout.supports(channels);
  }

  bool isVectorized() const { return m_layout.isVectorized(); }

  unsigned int vectorBlockSize() const { return m_layout.vectorBlockSize(); }

  static ActivationLayout promote(ActivationLayout layout,
                                  [[maybe_unused]] unsigned int channels) {
    switch (layout.kind()) {
    case ActivationLayoutKind::CHW:
      return ActivationLayout::CHW;
    case ActivationLayoutKind::HWC:
      if (channels % 8 == 0) {
        return ActivationLayout(ActivationLayoutKind::HWC8);
      } else {
        return ActivationLayout::HWC;
      }
    case ActivationLayoutKind::HWC8:
      return ActivationLayout(ActivationLayoutKind::HWC8);
    case ActivationLayoutKind::CHWC4:
      return ActivationLayout::CHWC4;
    case ActivationLayoutKind::CHWC8:
      return ActivationLayout::CHWC8;
    case ActivationLayoutKind::CHWC16:
      return ActivationLayout::CHWC16;
    default:
      diag::unreachable();
    }
  }

  static ActivationLayout demote(ActivationLayout layout,
                                 [[maybe_unused]] unsigned int channels) {
    switch (layout.kind()) {
    case ActivationLayoutKind::CHW:
      return ActivationLayout::CHW;
    case ActivationLayoutKind::HWC:
      return ActivationLayout::HWC;
    case ActivationLayoutKind::HWC8:
      return ActivationLayout::HWC;
    case ActivationLayoutKind::CHWC4:
      return ActivationLayout::CHWC4;
    case ActivationLayoutKind::CHWC8:
      return ActivationLayout::CHWC8;
    case ActivationLayoutKind::CHWC16:
      return ActivationLayout::CHWC16;
    default:
      diag::unreachable();
    }
  }

private:
  details::memory::tensors::ActivationLayout m_layout;
};

} // namespace denox::memory

template <>
struct fmt::formatter<
    denox::memory::details::memory::tensors::ActivationLayout> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(
      const denox::memory::details::memory::tensors::ActivationLayout &layout,
      FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{}", layout.to_string());
  }
};

template <>
struct fmt::formatter<denox::memory::ActivationLayout> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::ActivationLayout& layout,
              FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", layout.to_string());
  }
};
