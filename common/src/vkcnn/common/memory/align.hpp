#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
namespace vkcnn {

static inline std::size_t align_up(std::size_t offset,
                                   std::size_t alignment) noexcept {
  assert(alignment && (alignment & (alignment - 1)) == 0 &&
         "alignment must be power of two");
  return (offset + alignment - 1) & ~(alignment - 1);
}

static inline std::byte *align_up(std::byte *ptr,
                                  std::size_t alignment) noexcept {
  assert(alignment && (alignment & (alignment - 1)) == 0 &&
         "alignment must be power of two");
  auto p = reinterpret_cast<std::uintptr_t>(ptr);
  return reinterpret_cast<std::byte *>((p + alignment - 1) & ~(alignment - 1));
}

static inline const std::byte *align_up(const std::byte *ptr,
                                        std::size_t alignment) noexcept {
  assert(alignment && (alignment & (alignment - 1)) == 0 &&
         "alignment must be power of two");
  return const_cast<std::byte *>(
      align_up(const_cast<std::byte *>(ptr), alignment));
}

template <std::size_t Align>
static inline std::size_t align_up(std::size_t offset) noexcept {
  static_assert(Align && (Align & (Align - 1)) == 0 &&
                "alignment must be power of two");
  return (offset + Align - 1) & ~(Align - 1);
}

template <std::size_t Align>
static inline std::byte *align_up(std::byte *ptr) noexcept {
  static_assert(Align && (Align & (Align - 1)) == 0 &&
                "alignment must be power of two");
  auto p = reinterpret_cast<std::uintptr_t>(ptr);
  return reinterpret_cast<std::byte *>((p + Align - 1) & ~(Align - 1));
}

template <std::size_t Align>
static inline const std::byte *align_up(const std::byte *ptr) noexcept {
  static_assert(Align && (Align & (Align - 1)) == 0 &&
                "alignment must be power of two");
  return const_cast<std::byte *>(align_up<Align>(const_cast<std::byte *>(ptr)));
}

} // namespace vkcnn
