#pragma once

#include "diag/unreachable.hpp"
#include "memory/tensor/FilterShape.hpp"
#include <cassert>
#include <cstddef>

namespace denox::memory {

enum class FilterLayoutKind {
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

namespace details::memory::tensors {

class FilterLayout {
public:
  constexpr FilterLayout(FilterLayoutKind tag) : m_tag(tag) {}

  __attribute__((always_inline)) inline constexpr std::size_t
  operator()(const FilterShape &shape, unsigned int s, unsigned r,
             unsigned int c, unsigned int k) const {
    const std::size_t Rdim = static_cast<std::size_t>(shape.r);
    const std::size_t Sdim = static_cast<std::size_t>(shape.s);
    const std::size_t Cdim = static_cast<std::size_t>(shape.c);
    const std::size_t Kdim = static_cast<std::size_t>(shape.k);

    const std::size_t S = static_cast<std::size_t>(s);
    const std::size_t R = static_cast<std::size_t>(r);
    const std::size_t C = static_cast<std::size_t>(c);
    const std::size_t K = static_cast<std::size_t>(k);

    switch (m_tag) {
    case FilterLayoutKind::KCRS:
      return K * (Cdim * Rdim * Sdim) + C * (Rdim * Sdim) + R * Sdim + S;

    case FilterLayoutKind::KRSC:
      return K * (Rdim * Sdim * Cdim) + R * (Sdim * Cdim) + S * Cdim + C;

    case FilterLayoutKind::RSCK:
      return R * (Sdim * Cdim * Kdim) + S * (Cdim * Kdim) + C * Kdim + K;

    case FilterLayoutKind::RSKC:
      return R * (Sdim * Cdim * Kdim) + S * (Cdim * Kdim) + K * Cdim + C;

    case FilterLayoutKind::RSCKC8: {
      assert(Cdim % std::size_t{8} == 0);
      return R * (Sdim * Cdim * Kdim) + S * (Cdim * Kdim) +
             (C >> 3) * (Kdim << 3) + (K << 3) + (C & std::size_t{0x7});
    }

    case FilterLayoutKind::RCSKC8: {
      assert(Cdim % std::size_t{8} == 0);
      return R * (Cdim * Sdim * Kdim) + (C >> 3) * ((Sdim * Kdim) << 3) +
             S * (Kdim << 3) + (K << 3) + (C & std::size_t{0x7});
    }

    case FilterLayoutKind::RSCKC16: {
      assert(Cdim % std::size_t{16} == 0);
      return R * (Sdim * Cdim * Kdim) + S * (Cdim * Kdim) +
             (C >> 4) * (Kdim << 4) + (K << 4) + (C & std::size_t{0xF});
    }

    case FilterLayoutKind::RCSKC16: {
      assert(Cdim % std::size_t{16} == 0);
      return R * (Cdim * Sdim * Kdim) + (C >> 4) * ((Sdim * Kdim) << 4) +
             S * (Kdim << 4) + (K << 4) + (C & std::size_t{0xF});
    }

    case FilterLayoutKind::RSKCK8: {
      assert(Kdim % std::size_t{8} == 0);
      return R * (Cdim * Sdim * Kdim) + S * (Cdim * Kdim) +
             (K >> 3) * (Cdim << 3) + (C << 3) + (K & std::size_t{0x7});
    }

    case FilterLayoutKind::RSKCK16: {
      assert(Kdim % std::size_t{16} == 0);
      return R * (Cdim * Sdim * Kdim) + S * (Cdim * Kdim) +
             (K >> 4) * (Cdim << 4) + (C << 4) + (K & std::size_t{0xF});
    }

    case FilterLayoutKind::KRSCK8: {
      assert(Kdim % std::size_t{8} == 0);
      return (K >> 3) * ((Rdim * Sdim * Cdim) << 3) + R * ((Sdim * Cdim) << 3) +
             S * (Cdim << 3) + (C << 3) + (K & std::size_t{0x7});
    }

    case FilterLayoutKind::KRSCK16: {
      assert(Kdim % std::size_t{16} == 0);
      return (K >> 4) * ((Rdim * Sdim * Cdim) << 4) + R * ((Sdim * Cdim) << 4) +
             S * (Cdim << 4) + (C << 4) + (K & std::size_t{0xF});
    }
    }
    denox::compiler::diag::unreachable();
  }

  friend bool operator==(const FilterLayout &lhs, const FilterLayout &rhs) {
    return lhs.m_tag == rhs.m_tag;
  }
  friend bool operator!=(const FilterLayout &lhs, const FilterLayout &rhs) {
    return lhs.m_tag != rhs.m_tag;
  }

  FilterLayoutKind kind() const { return m_tag; }

private:
  FilterLayoutKind m_tag;
};

} // namespace details::memory::tensors

class FilterLayout {
public:
  constexpr FilterLayout(details::memory::tensors::FilterLayout layout)
      : m_layout(layout) {}

  static constexpr details::memory::tensors::FilterLayout KRSC{
      FilterLayoutKind::KRSC};

  static constexpr details::memory::tensors::FilterLayout KCRS{
      FilterLayoutKind::KCRS};

  static constexpr details::memory::tensors::FilterLayout RSCK{
      FilterLayoutKind::RSCK};

  static constexpr details::memory::tensors::FilterLayout RSKC{
      FilterLayoutKind::RSKC};

  static constexpr details::memory::tensors::FilterLayout RSCKC8{
      FilterLayoutKind::RSCKC8};

  static constexpr details::memory::tensors::FilterLayout RCSKC8{
      FilterLayoutKind::RCSKC8};

  static constexpr details::memory::tensors::FilterLayout RSCKC16{
      FilterLayoutKind::RSCKC16};

  static constexpr details::memory::tensors::FilterLayout RCSKC16{
      FilterLayoutKind::RCSKC16};

  static constexpr details::memory::tensors::FilterLayout RSKCK8{
      FilterLayoutKind::RSKCK8};

  static constexpr details::memory::tensors::FilterLayout RSKCK16{
      FilterLayoutKind::RSKCK16};

  static constexpr details::memory::tensors::FilterLayout KRSCK8{
      FilterLayoutKind::KRSCK8};

  static constexpr details::memory::tensors::FilterLayout KRSCK16{
      FilterLayoutKind::KRSCK16};

  constexpr std::size_t operator()(const FilterShape &shape, unsigned int s,
                                   unsigned r, unsigned int c,
                                   unsigned int k) const {
    return m_layout(shape, s, r, c, k);
  }
  FilterLayoutKind kind() const { return m_layout.kind(); }

  friend bool operator==(const FilterLayout &lhs, const FilterLayout &rhs) {
    return lhs.m_layout == rhs.m_layout;
  }
  friend bool operator!=(const FilterLayout &lhs, const FilterLayout &rhs) {
    return lhs.m_layout != rhs.m_layout;
  }

private:
  details::memory::tensors::FilterLayout m_layout;
};
} // namespace denox::compiler
