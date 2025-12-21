#pragma once

#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/dtype/dtype_reference.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FitlerDescriptor.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <fmt/core.h>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>

namespace denox::memory {

// fwd declarations
class FilterTensor;
class FilterTensorConstView;

class FilterTensorView {
public:
  friend class FilterTensorConstView;
  explicit FilterTensorView(FilterDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit FilterTensorView(FilterTensor *tensor);
  FilterTensorView(FilterTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span<const std::byte>{m_buffer, m_desc.byteSize()};
  }
  std::span<std::byte> span() {
    return std::span<std::byte>{m_buffer, m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int s, unsigned int r, unsigned int c,
                     unsigned int k) {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  dtype_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                           unsigned int k) const {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  dtype_reference operator[](unsigned int linearIndex) {
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  dtype_const_reference operator[](unsigned int linearIndex) const {
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  void assignFrom(const FilterTensorConstView &view);

private:
  FilterDescriptor m_desc;
  std::byte *m_buffer;
};

class FilterTensorConstView {
public:
  explicit FilterTensorConstView(FilterDescriptor desc, const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  FilterTensorConstView(FilterTensorView view)
      : m_desc(view.m_desc), m_buffer(view.m_buffer) {}

  explicit FilterTensorConstView(const FilterTensor *tensor);
  FilterTensorConstView(const FilterTensor &tensor);

  const std::byte *data() const { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span<const std::byte>{m_buffer, m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                           unsigned int k) const {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  dtype_const_reference operator[](unsigned int linearIndex) const {
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

private:
  FilterDescriptor m_desc;
  const std::byte *m_buffer;
};

class FilterTensor {
private:
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterTensor(FilterDescriptor descriptor,
                        std::span<const std::byte> weights,
                        const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    assert(n == weights.size());
    std::memcpy(ptr, weights.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
    requires(!std::same_as<Alloc, std::span<std::byte>> &&
             !std::same_as<Alloc, FilterTensorConstView>)
  explicit FilterTensor(FilterDescriptor descriptor, const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memset(ptr, 0, n);
    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterTensor(FilterDescriptor descriptor, FilterTensorConstView view,
                        const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    if (shape() != view.shape()) {
      throw std::runtime_error("Invalid tensor shape! Shapes do not match!");
    }
    if (layout() == view.layout() && type() == view.type()) {
      std::memcpy(ptr, view.data(), n);
    } else {
      FilterTensorView{m_desc, ptr}.assignFrom(view);
    }

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterTensor(FilterTensorConstView view, const Alloc &alloc = {})
      : m_desc(view.desc()) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memcpy(ptr, view.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  const std::byte *data() const { return m_storage.get(); }
  std::byte *data() { return m_storage.get(); }

  std::span<const std::byte> span() const {
    return std::span<const std::byte>{m_storage.get(), m_desc.byteSize()};
  }
  std::span<std::byte> span() {
    return std::span<std::byte>{m_storage.get(), m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int s, unsigned int r, unsigned int c,
                     unsigned int k) {
    return FilterTensorView{this}.at(s, r, c, k);
  }

  dtype_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                           unsigned int k) const {
    return FilterTensorConstView{this}.at(s, r, c, k);
  }

  dtype_reference operator[](unsigned int linearIndex) {
    return FilterTensorView{this}[linearIndex];
  }

  dtype_const_reference operator[](unsigned int linearIndex) const {
    return FilterTensorConstView{this}[linearIndex];
  }

private:
  FilterDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::FilterTensor> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::FilterTensor &t, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{desc={}, bytes={}}}", t.desc(),
                          t.byteSize());
  }
};

template <>
struct fmt::formatter<denox::memory::FilterTensorView> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::FilterTensorView& t,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "{{desc={}, bytes={}}}",
        t.desc(),
        t.byteSize());
  }
};

template <>
struct fmt::formatter<denox::memory::FilterTensorConstView> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::memory::FilterTensorConstView& t,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "{{desc={}, bytes={}}}",
        t.desc(),
        t.byteSize());
  }
};
