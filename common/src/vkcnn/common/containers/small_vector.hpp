#pragma once
#include <cassert>
#include <cstddef>
#include <cstddef> // std::byte
#include <initializer_list>
#include <iterator>
#include <memory> // std::construct_at, std::destroy_at
#include <new>
#include <type_traits>
#include <utility>

namespace vkcnn::containers {

template <class T, std::size_t N> class small_vector {
  static_assert(N > 0, "N must be > 0");

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

  small_vector() noexcept : m_ptr(inline_ptr()), m_size(0), m_capacity(N) {}

  explicit small_vector(size_type count) : small_vector() { resize(count); }

  small_vector(size_type count, const T &value) : small_vector() {
    resize(count, value);
  }

  template <class It> small_vector(It first, It last) : small_vector() {
    assign(first, last);
  }

  small_vector(std::initializer_list<T> il)
      : small_vector(il.begin(), il.end()) {}

  small_vector(const small_vector &other) : small_vector() {
    reserve(other.m_size);
    uninitialized_copy_n(other.m_ptr, other.m_size, m_ptr);
    m_size = other.m_size;
  }

  small_vector(small_vector &&other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : m_ptr(inline_ptr()), m_size(0), m_capacity(N) {
    move_from(std::move(other));
  }

  ~small_vector() {
    destroy_range(m_ptr, m_size);
    if (!is_small())
      operator delete[](m_ptr, std::align_val_t{alignof(T)});
  }

  small_vector &operator=(const small_vector &rhs) {
    if (this == &rhs)
      return *this;
    if (rhs.m_size <= m_capacity) {
      // reuse buffer
      const size_type common = m_size < rhs.m_size ? m_size : rhs.m_size;
      for (size_type i = 0; i < common; ++i)
        m_ptr[i] = rhs.m_ptr[i];
      if (m_size < rhs.m_size) {
        uninitialized_copy_n(rhs.m_ptr + m_size, rhs.m_size - m_size,
                             m_ptr + m_size);
      } else {
        destroy_range(m_ptr + rhs.m_size, m_size - rhs.m_size);
      }
      m_size = rhs.m_size;
    } else {
      pointer new_buf = allocate(rhs.m_size);
      try {
        uninitialized_copy_n(rhs.m_ptr, rhs.m_size, new_buf);
      } catch (...) {
        operator delete[](new_buf, std::align_val_t{alignof(T)});
        throw;
      }
      destroy_range(m_ptr, m_size);
      if (!is_small())
        operator delete[](m_ptr, std::align_val_t{alignof(T)});
      m_ptr = new_buf;
      m_size = rhs.m_size;
      m_capacity = rhs.m_size;
    }
    return *this;
  }

  small_vector &operator=(small_vector &&rhs) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    if (this == &rhs)
      return *this;
    clear_and_release();
    move_from(std::move(rhs));
    return *this;
  }

  small_vector &operator=(std::initializer_list<T> il) {
    assign(il.begin(), il.end());
    return *this;
  }

  [[nodiscard]] size_type size() const noexcept { return m_size; }
  [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
  [[nodiscard]] size_type capacity() const noexcept { return m_capacity; }

  void reserve(size_type new_cap) {
    if (new_cap <= m_capacity)
      return;
    reallocate_(grow_to(new_cap));
  }

  void shrink_to_fit() {
    if (is_small())
      return;
    if (m_size == 0) {
      clear_and_release();
      return;
    }
    if (m_size <= N) {
      pointer dst = inline_ptr();
      uninitialized_move_n(m_ptr, m_size, dst);
      destroy_range(m_ptr, m_size);
      operator delete[](m_ptr, std::align_val_t{alignof(T)});
      m_ptr = dst;
      m_capacity = N;
    } else if (m_size < m_capacity) {
      pointer dst = allocate(m_size);
      uninitialized_move_n(m_ptr, m_size, dst);
      destroy_range(m_ptr, m_size);
      operator delete[](m_ptr, std::align_val_t{alignof(T)});
      m_ptr = dst;
      m_capacity = m_size;
    }
  }

  void clear() noexcept {
    destroy_range(m_ptr, m_size);
    m_size = 0;
  }

  void resize(size_type n) {
    if (n < m_size) {
      destroy_range(m_ptr + n, m_size - n);
      m_size = n;
    } else if (n > m_size) {
      reserve(n);
      for (; m_size < n; ++m_size)
        std::construct_at(m_ptr + m_size);
    }
  }

  void resize(size_type n, const T &value) {
    if (n < m_size) {
      destroy_range(m_ptr + n, m_size - n);
      m_size = n;
    } else if (n > m_size) {
      reserve(n);
      for (; m_size < n; ++m_size)
        std::construct_at(m_ptr + m_size, value);
    }
  }

  reference operator[](size_type i) noexcept { return m_ptr[i]; }
  const_reference operator[](size_type i) const noexcept { return m_ptr[i]; }
  reference front() noexcept { return m_ptr[0]; }
  const_reference front() const noexcept { return m_ptr[0]; }
  reference back() noexcept { return m_ptr[m_size - 1]; }
  const_reference back() const noexcept { return m_ptr[m_size - 1]; }
  pointer data() noexcept { return m_ptr; }
  const_pointer data() const noexcept { return m_ptr; }

  iterator begin() noexcept { return m_ptr; }
  const_iterator begin() const noexcept { return m_ptr; }
  const_iterator cbegin() const noexcept { return m_ptr; }
  iterator end() noexcept { return m_ptr + m_size; }
  const_iterator end() const noexcept { return m_ptr + m_size; }
  const_iterator cend() const noexcept { return m_ptr + m_size; }

  void push_back(const T &v) {
    ensure_capacity(m_size + 1);
    std::construct_at(m_ptr + m_size, v);
    ++m_size;
  }

  void push_back(T &&v) {
    ensure_capacity(m_size + 1);
    std::construct_at(m_ptr + m_size, std::move(v));
    ++m_size;
  }

  template <class... Args> reference emplace_back(Args &&...args) {
    ensure_capacity(m_size + 1);
    std::construct_at(m_ptr + m_size, std::forward<Args>(args)...);
    return m_ptr[m_size++];
  }

  void pop_back() noexcept {
    assert(m_size > 0);
    --m_size;
    std::destroy_at(m_ptr + m_size);
  }

  iterator insert(const_iterator cpos, const T &value) {
    return emplace(cpos, value);
  }
  iterator insert(const_iterator cpos, T &&value) {
    return emplace(cpos, std::move(value));
  }

  template <class... Args>
  iterator emplace(const_iterator cpos, Args &&...args) {
    size_type idx = static_cast<size_type>(cpos - cbegin());
    ensure_capacity(m_size + 1);

    if (idx == m_size) {
      std::construct_at(m_ptr + m_size, std::forward<Args>(args)...);
      ++m_size;
      return m_ptr + idx;
    }

    std::construct_at(m_ptr + m_size, std::move(m_ptr[m_size - 1]));
    for (size_type i = m_size - 1; i > idx; --i) {
      m_ptr[i] = std::move(m_ptr[i - 1]);
    }
    std::destroy_at(m_ptr + idx);
    std::construct_at(m_ptr + idx, std::forward<Args>(args)...);
    ++m_size;
    return m_ptr + idx;
  }

  iterator erase(const_iterator cpos) {
    size_type idx = static_cast<size_type>(cpos - cbegin());
    assert(idx < m_size);
    for (size_type i = idx; i + 1 < m_size; ++i) {
      m_ptr[i] = std::move(m_ptr[i + 1]);
    }
    --m_size;
    std::destroy_at(m_ptr + m_size);
    return m_ptr + idx;
  }

  iterator erase(const_iterator first, const_iterator last) {
    size_type i = static_cast<size_type>(first - cbegin());
    size_type j = static_cast<size_type>(last - cbegin());
    assert(i <= j && j <= m_size);
    size_type count = j - i;
    if (count == 0)
      return m_ptr + i;

    for (size_type k = i; k + count < m_size; ++k) {
      m_ptr[k] = std::move(m_ptr[k + count]);
    }
    destroy_range(m_ptr + (m_size - count), count);
    m_size -= count;
    return m_ptr + i;
  }

  // --- assign helpers ---
  template <class It> void assign(It first, It last) {
    clear();
    if constexpr (std::forward_iterator<It>) {
      size_type n = static_cast<size_type>(std::distance(first, last));
      reserve(n);
      for (; first != last; ++first) {
        std::construct_at(m_ptr + m_size, *first);
        ++m_size;
      }
    } else {
      for (; first != last; ++first) {
        push_back(*first);
      }
    }
  }

  void assign(size_type count, const T &value) {
    clear();
    reserve(count);
    for (size_type i = 0; i < count; ++i) {
      std::construct_at(m_ptr + m_size, value);
      ++m_size;
    }
  }

private:
  struct SmallStorage {
    alignas(T) std::byte data[sizeof(T)];
  };
  SmallStorage m_smallStorage[N];
  pointer m_ptr;
  size_type m_size;
  size_type m_capacity;

  [[nodiscard]] pointer inline_ptr() noexcept {
    return std::launder(reinterpret_cast<pointer>(m_smallStorage));
  }
  [[nodiscard]] bool is_small() const noexcept {
    return m_ptr == const_cast<small_vector *>(this)->inline_ptr();
  }

  static pointer allocate(size_type n) {
    if (n == 0)
      return nullptr;
    return static_cast<pointer>(operator new[](n * sizeof(T),
                                               std::align_val_t{alignof(T)}));
  }

  static void uninitialized_copy_n(const_pointer src, size_type n,
                                   pointer dst) {
    size_type i = 0;
    try {
      for (; i < n; ++i)
        std::construct_at(dst + i, src[i]);
    } catch (...) {
      for (size_type k = 0; k < i; ++k)
        std::destroy_at(dst + k);
      throw;
    }
  }

  static void uninitialized_move_n(pointer src, size_type n, pointer dst) {
    size_type i = 0;
    try {
      for (; i < n; ++i)
        std::construct_at(dst + i, std::move(src[i]));
    } catch (...) {
      for (size_type k = 0; k < i; ++k)
        std::destroy_at(dst + k);
      throw;
    }
  }

  static void destroy_range(pointer p, size_type n) noexcept {
    for (size_type i = 0; i < n; ++i)
      std::destroy_at(p + i);
  }

  void clear_and_release() noexcept {
    destroy_range(m_ptr, m_size);
    if (!is_small())
      operator delete[](m_ptr, std::align_val_t{alignof(T)});
    m_ptr = inline_ptr();
    m_size = 0;
    m_capacity = N;
  }

  [[nodiscard]] size_type grow_to(size_type new_cap) const {
    size_type g = m_capacity * 2;
    if (g < new_cap)
      g = new_cap;
    return g;
  }

  void ensure_capacity(size_type need) {
    if (need <= m_capacity)
      return;
    reallocate_(grow_to(need));
  }

  void reallocate_(size_type new_cap) {
    pointer new_buf = allocate(new_cap);
    try {
      uninitialized_move_n(m_ptr, m_size, new_buf);
    } catch (...) {
      operator delete[](new_buf, std::align_val_t{alignof(T)});
      throw;
    }
    destroy_range(m_ptr, m_size);
    if (!is_small())
      operator delete[](m_ptr, std::align_val_t{alignof(T)});
    m_ptr = new_buf;
    m_capacity = new_cap;
  }

  void move_from(small_vector &&other) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    if (!other.is_small()) {
      m_ptr = other.m_ptr;
      m_size = other.m_size;
      m_capacity = other.m_capacity;
      other.m_ptr = other.inline_ptr();
      other.m_size = 0;
      other.m_capacity = N;
    } else {
      uninitialized_move_n(other.m_ptr, other.m_size, m_ptr);
      m_size = other.m_size;
      destroy_range(other.m_ptr, other.m_size);
      other.m_size = 0;
    }
  }
};

} // namespace vkcnn
