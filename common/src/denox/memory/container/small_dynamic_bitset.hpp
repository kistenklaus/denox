#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace denox::memory {

// Small-buffer-optimized dynamic bitset.
// Template parameter Small is the number of *bits* stored inline.
template <std::size_t Small>
class small_dynamic_bitset {
public:
  using word_type = std::uint64_t;
  using size_type = std::size_t;

  static constexpr size_type word_bits = 64;
  static constexpr size_type npos = std::numeric_limits<size_type>::max();
  static constexpr size_type small_words = (Small + word_bits - 1) / word_bits;

  small_dynamic_bitset() noexcept { init_empty(); }

  explicit small_dynamic_bitset(size_type bit_count, bool value = false) {
    init_empty();
    resize(bit_count, value);
  }

  small_dynamic_bitset(const small_dynamic_bitset& other) {
    init_empty();
    assign_from(other);
  }

  small_dynamic_bitset(small_dynamic_bitset&& other) noexcept {
    init_empty();
    move_from(std::move(other));
  }

  small_dynamic_bitset& operator=(const small_dynamic_bitset& other) {
    if (this != &other) assign_from(other);
    return *this;
  }

  small_dynamic_bitset& operator=(small_dynamic_bitset&& other) noexcept {
    if (this != &other) {
      destroy();
      init_empty();
      move_from(std::move(other));
    }
    return *this;
  }

  ~small_dynamic_bitset() { destroy(); }

  // ---- Capacity / Size ----
  size_type size() const noexcept { return m_bits; }
  bool empty() const noexcept { return m_bits == 0; }

  void clear() noexcept { resize(0); }

  void resize(size_type bit_count, bool value = false) {
    const size_type new_words = words_for_bits(bit_count);

    // Reallocate if needed
    if (needs_heap(new_words)) {
      if (!m_heap || new_words != m_words || m_storage != storage_kind::heap) {
        // allocate new heap buffer
        word_type* new_buf = static_cast<word_type*>(std::malloc(new_words * sizeof(word_type)));
        if (!new_buf) std::abort();

        // initialize
        std::fill_n(new_buf, new_words, word_type{0});

        // copy old content
        const size_type copy_words = std::min(m_words, new_words);
        if (copy_words > 0) {
          std::memcpy(new_buf, data(), copy_words * sizeof(word_type));
        }

        // free old heap if any
        if (m_storage == storage_kind::heap && m_heap) {
          std::free(m_heap);
          m_heap = nullptr;
        }

        m_heap = new_buf;
        m_storage = storage_kind::heap;
      }
    } else {
      // Switch to small storage if necessary
      if (m_storage == storage_kind::heap) {
        // copy from heap to small, then free heap
        const size_type copy_words = std::min(m_words, new_words);
        std::fill_n(m_small, small_words, word_type{0});
        if (copy_words > 0) {
          std::memcpy(m_small, m_heap, copy_words * sizeof(word_type));
        }
        std::free(m_heap);
        m_heap = nullptr;
        m_storage = storage_kind::small;
      }
      // Ensure words beyond new_words are cleared in small buffer
      if (m_storage == storage_kind::small) {
        for (size_type i = new_words; i < small_words; ++i) m_small[i] = 0;
      }
    }

    // If growing and value==true, set new bits.
    const size_type old_bits = m_bits;
    m_bits = bit_count;
    m_words = new_words;

    if (bit_count > old_bits && value) {
      set_range(old_bits, bit_count);
    }

    trim_tail_bits();
  }

  // ---- Element access ----
  bool test(size_type pos) const {
    assert(pos < m_bits);
    const size_type w = pos / word_bits;
    const size_type b = pos % word_bits;
    return (data()[w] >> b) & word_type{1};
  }

  bool operator[](size_type pos) const { return test(pos); }

  void set(size_type pos, bool value = true) {
    assert(pos < m_bits);
    const size_type w = pos / word_bits;
    const size_type b = pos % word_bits;
    word_type& ww = data()[w];
    const word_type mask = word_type{1} << b;
    if (value) ww |= mask;
    else ww &= ~mask;
  }

  void reset(size_type pos) { set(pos, false); }

  void flip(size_type pos) {
    assert(pos < m_bits);
    const size_type w = pos / word_bits;
    const size_type b = pos % word_bits;
    data()[w] ^= (word_type{1} << b);
    trim_tail_bits();
  }

  void reset_all() {
    std::fill_n(data(), m_words, word_type{0});
  }

  void set_all() {
    std::fill_n(data(), m_words, ~word_type{0});
    trim_tail_bits();
  }

  void flip_all() {
    for (size_type i = 0; i < m_words; ++i) data()[i] = ~data()[i];
    trim_tail_bits();
  }

  // ---- Queries ----
  bool any() const {
    for (size_type i = 0; i < m_words; ++i) {
      if (data()[i] != 0) return true;
    }
    return false;
  }

  bool none() const { return !any(); }

  size_type count() const {
    size_type c = 0;
    for (size_type i = 0; i < m_words; ++i) {
#if defined(__GNUG__) || defined(__clang__)
      c += static_cast<size_type>(__builtin_popcountll(static_cast<unsigned long long>(data()[i])));
#else
      // Portable popcount
      word_type x = data()[i];
      while (x) { x &= (x - 1); ++c; }
#endif
    }
    return c;
  }

  // Find first set bit at or after `pos` (pos can be 0..size). Returns npos if none.
  size_type find_next(size_type pos = 0) const {
    if (pos >= m_bits) return npos;

    size_type w = pos / word_bits;
    size_type b = pos % word_bits;

    word_type x = data()[w];
    x &= (~word_type{0} << b);

    while (true) {
      if (x != 0) {
#if defined(__GNUG__) || defined(__clang__)
        const int tz = __builtin_ctzll(static_cast<unsigned long long>(x));
        const size_type idx = w * word_bits + static_cast<size_type>(tz);
        return (idx < m_bits) ? idx : npos;
#else
        // Portable ctz
        size_type tz = 0;
        while (((x >> tz) & 1ull) == 0ull) ++tz;
        const size_type idx = w * word_bits + tz;
        return (idx < m_bits) ? idx : npos;
#endif
      }
      ++w;
      if (w >= m_words) break;
      x = data()[w];
    }
    return npos;
  }

  // ---- Bitwise operators (same size required) ----
  small_dynamic_bitset& operator&=(const small_dynamic_bitset& rhs) {
    assert(m_bits == rhs.m_bits);
    for (size_type i = 0; i < m_words; ++i) data()[i] &= rhs.data()[i];
    return *this;
  }

  small_dynamic_bitset& operator|=(const small_dynamic_bitset& rhs) {
    assert(m_bits == rhs.m_bits);
    for (size_type i = 0; i < m_words; ++i) data()[i] |= rhs.data()[i];
    return *this;
  }

  small_dynamic_bitset& operator^=(const small_dynamic_bitset& rhs) {
    assert(m_bits == rhs.m_bits);
    for (size_type i = 0; i < m_words; ++i) data()[i] ^= rhs.data()[i];
    trim_tail_bits();
    return *this;
  }

  friend small_dynamic_bitset operator&(small_dynamic_bitset lhs, const small_dynamic_bitset& rhs) {
    lhs &= rhs;
    return lhs;
  }

  friend small_dynamic_bitset operator|(small_dynamic_bitset lhs, const small_dynamic_bitset& rhs) {
    lhs |= rhs;
    return lhs;
  }

  friend small_dynamic_bitset operator^(small_dynamic_bitset lhs, const small_dynamic_bitset& rhs) {
    lhs ^= rhs;
    return lhs;
  }

  friend small_dynamic_bitset operator~(small_dynamic_bitset x) {
    x.flip_all();
    return x;
  }

  // ---- Shifts ----
  small_dynamic_bitset& operator<<=(size_type shift) {
    if (shift == 0 || m_bits == 0) return *this;
    if (shift >= m_bits) { reset_all(); return *this; }

    const size_type word_shift = shift / word_bits;
    const size_type bit_shift  = shift % word_bits;

    if (word_shift) {
      for (size_type i = m_words; i-- > 0;) {
        data()[i] = (i >= word_shift) ? data()[i - word_shift] : word_type{0};
      }
    }

    if (bit_shift) {
      for (size_type i = m_words; i-- > 0;) {
        const word_type hi = data()[i] << bit_shift;
        const word_type lo = (i > 0) ? (data()[i - 1] >> (word_bits - bit_shift)) : word_type{0};
        data()[i] = hi | lo;
      }
    }

    trim_tail_bits();
    return *this;
  }

  small_dynamic_bitset& operator>>=(size_type shift) {
    if (shift == 0 || m_bits == 0) return *this;
    if (shift >= m_bits) { reset_all(); return *this; }

    const size_type word_shift = shift / word_bits;
    const size_type bit_shift  = shift % word_bits;

    if (word_shift) {
      for (size_type i = 0; i < m_words; ++i) {
        data()[i] = (i + word_shift < m_words) ? data()[i + word_shift] : word_type{0};
      }
    }

    if (bit_shift) {
      for (size_type i = 0; i < m_words; ++i) {
        const word_type lo = data()[i] >> bit_shift;
        const word_type hi = (i + 1 < m_words) ? (data()[i + 1] << (word_bits - bit_shift)) : word_type{0};
        data()[i] = lo | hi;
      }
    }

    trim_tail_bits();
    return *this;
  }

  friend small_dynamic_bitset operator<<(small_dynamic_bitset lhs, size_type s) { lhs <<= s; return lhs; }
  friend small_dynamic_bitset operator>>(small_dynamic_bitset lhs, size_type s) { lhs >>= s; return lhs; }

  // ---- Equality ----
  friend bool operator==(const small_dynamic_bitset& a, const small_dynamic_bitset& b) {
    if (a.m_bits != b.m_bits) return false;
    for (size_type i = 0; i < a.m_words; ++i) {
      if (a.data()[i] != b.data()[i]) return false;
    }
    return true;
  }
  friend bool operator!=(const small_dynamic_bitset& a, const small_dynamic_bitset& b) { return !(a == b); }

  // ---- Low-level access (optional) ----
  const word_type* words() const noexcept { return data(); }
  word_type* words() noexcept { return data(); }
  size_type word_count() const noexcept { return m_words; }

private:
  enum class storage_kind : std::uint8_t { small, heap };

  size_type m_bits = 0;
  size_type m_words = 0;
  storage_kind m_storage = storage_kind::small;

  word_type m_small[small_words > 0 ? small_words : 1]{}; // Small==0 still compiles
  word_type* m_heap = nullptr;

  static constexpr size_type words_for_bits(size_type bits) noexcept {
    return (bits + word_bits - 1) / word_bits;
  }

  static constexpr bool needs_heap(size_type words) noexcept {
    return words > small_words;
  }

  word_type* data() noexcept {
    return (m_storage == storage_kind::small) ? m_small : m_heap;
  }
  const word_type* data() const noexcept {
    return (m_storage == storage_kind::small) ? m_small : m_heap;
  }

  void init_empty() noexcept {
    m_bits = 0;
    m_words = 0;
    m_storage = storage_kind::small;
    m_heap = nullptr;
    if constexpr (small_words > 0) {
      std::fill_n(m_small, small_words, word_type{0});
    }
  }

  void destroy() noexcept {
    if (m_storage == storage_kind::heap && m_heap) {
      std::free(m_heap);
      m_heap = nullptr;
    }
    m_bits = 0;
    m_words = 0;
    m_storage = storage_kind::small;
  }

  void assign_from(const small_dynamic_bitset& other) {
    resize(other.m_bits, false);
    if (m_words) std::memcpy(data(), other.data(), m_words * sizeof(word_type));
  }

  void move_from(small_dynamic_bitset&& other) noexcept {
    m_bits = other.m_bits;
    m_words = other.m_words;
    m_storage = other.m_storage;

    if (other.m_storage == storage_kind::small) {
      if constexpr (small_words > 0) {
        std::memcpy(m_small, other.m_small, small_words * sizeof(word_type));
      }
      m_heap = nullptr;
    } else {
      m_heap = other.m_heap;
      other.m_heap = nullptr;
    }

    other.init_empty();
  }

  void trim_tail_bits() {
    if (m_bits == 0 || m_words == 0) return;
    const size_type tail = m_bits % word_bits;
    if (tail == 0) return;
    const word_type mask = (tail == 64) ? ~word_type{0} : ((word_type{1} << tail) - 1);
    data()[m_words - 1] &= mask;
  }

  void set_range(size_type from_bit, size_type to_bit) {
    // set bits in [from_bit, to_bit)
    if (from_bit >= to_bit) return;
    assert(to_bit <= m_bits);

    size_type i = from_bit;
    while (i < to_bit && (i % word_bits) != 0) {
      set(i, true);
      ++i;
    }

    while (i + word_bits <= to_bit) {
      data()[i / word_bits] = ~word_type{0};
      i += word_bits;
    }

    while (i < to_bit) {
      set(i, true);
      ++i;
    }
  }
};

} // namespace denox::memory
