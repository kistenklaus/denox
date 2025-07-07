#pragma once

#include "vkcnn/common/tensor/FilterLayout.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
namespace vkcnn {

// fwd declarations
class FilterHostTensor;
class FilterHostTensorConstView;

class FilterHostTensorView {
public:
  friend class FilterHostTensorConstView;
  explicit FilterHostTensorView(FilterDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit FilterHostTensorView(FilterHostTensor *tensor);
  FilterHostTensorView(FilterHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }
  std::span<std::byte> span() { return std::span{m_buffer, m_desc.byteSize()}; }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType floatType() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  template <typename F>
  auto at(unsigned int s, unsigned int r, unsigned int c, unsigned int k)
      -> std::conditional_t<std::same_as<std::remove_cvref_t<F>, f16>,
                            f16_reference, std::remove_cvref_t<F> &> {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    if constexpr (std::same_as<std::remove_cvref_t<F>, f16>) {
      return f16_reference(reinterpret_cast<f16 *>(ptr));
    } else {
      return *reinterpret_cast<std::remove_cvref_t<F> *>(ptr);
    }
  }

  template <typename F>
  auto at(unsigned int s, unsigned int r, unsigned int c, unsigned int k) const
      -> std::conditional_t<std::same_as<std::remove_cvref_t<F>, f16>,
                            f16_const_reference,
                            const std::remove_cvref_t<F> &> {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    if constexpr (std::same_as<std::remove_cvref_t<F>, f16>) {
      return f16_const_reference(reinterpret_cast<const f16 *>(ptr));
    } else {
      return *reinterpret_cast<const std::remove_cvref_t<F> *>(ptr);
    }
  }

  void assignFrom(const FilterHostTensorConstView &view);

private:
  FilterDescriptor m_desc;
  std::byte *m_buffer;
};

class FilterHostTensorConstView {
public:
  explicit FilterHostTensorConstView(FilterDescriptor desc,
                                     const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  FilterHostTensorConstView(FilterHostTensorView view)
      : m_desc(view.m_desc), m_buffer(view.m_buffer) {}

  explicit FilterHostTensorConstView(const FilterHostTensor *tensor);
  FilterHostTensorConstView(const FilterHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType floatType() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  template <typename F>
  auto at(unsigned int s, unsigned int r, unsigned int c, unsigned int k) const
      -> std::conditional_t<std::same_as<std::remove_cvref_t<F>, f16>,
                            f16_const_reference,
                            const std::remove_cvref_t<F> &> {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    if constexpr (std::same_as<std::remove_cvref_t<F>, f16>) {
      return f16_const_reference(reinterpret_cast<const f16 *>(ptr));
    } else {
      return *reinterpret_cast<const std::remove_cvref_t<F> *>(ptr);
    }
  }

private:
  FilterDescriptor m_desc;
  const std::byte *m_buffer;
};

class FilterHostTensor {
private:
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterHostTensor(FilterDescriptor descriptor,
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
  explicit FilterHostTensor(FilterDescriptor descriptor,
                            const Alloc &alloc = {})
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
  explicit FilterHostTensor(FilterDescriptor descriptor,
                            FilterHostTensorConstView view,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    if (shape() != view.shape()) {
      throw std::runtime_error("Invalid tensor shape! Shapes do not match!");
    }
    if (layout() == view.layout() && floatType() == view.floatType()) {
      std::memcpy(ptr, view.data(), n);
    } else {
      FilterHostTensorView{m_desc, ptr}.assignFrom(view);
    }

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterHostTensor(FilterHostTensorConstView view,
                            const Alloc &alloc = {})
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
    return std::span{m_storage.get(), m_desc.byteSize()};
  }
  std::span<std::byte> span() {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType floatType() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  template <typename F>
  auto at(unsigned int s, unsigned int r, unsigned int c, unsigned int k)
      -> std::conditional_t<std::same_as<std::remove_cvref_t<F>, f16>,
                            f16_reference, std::remove_cvref_t<F> &> {
    return FilterHostTensorView{this}.at<F>(s, r, c, k);
  }

  template <typename F>
  auto at(unsigned int s, unsigned int r, unsigned int c, unsigned int k) const
      -> std::conditional_t<std::same_as<std::remove_cvref_t<F>, f16>,
                            f16_const_reference,
                            const std::remove_cvref_t<F> &> {
    return FilterHostTensorConstView{this}.at<F>(s, r, c, k);
  }

private:
  FilterDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace vkcnn
