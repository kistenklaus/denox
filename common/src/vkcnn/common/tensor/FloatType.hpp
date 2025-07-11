#pragma once

#include <bit>
#include <compare>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <utility>

namespace vkcnn {

namespace details {
class FloatType {
public:
  enum Tag {
    F16,
    F32,
    F64,
  };
  constexpr FloatType(Tag tag) : m_tag(tag) {}

  constexpr inline std::size_t size() const {
    switch (m_tag) {
    case F16:
      return 2;
    case F32:
      return 4;
    case F64:
      return 8;
    }
    throw std::runtime_error("Unsupported FloatType");
  }

  constexpr auto operator<=>(const FloatType &) const = default;

private:
  Tag m_tag;
};
} // namespace details

class FloatType {
public:
  constexpr FloatType(details::FloatType type) : m_type(type) {}

  static constexpr details::FloatType F16{details::FloatType::F16};
  static constexpr details::FloatType F32{details::FloatType::F32};
  static constexpr details::FloatType F64{details::FloatType::F64};

  std::size_t size() const { return m_type.size(); }

  constexpr auto operator<=>(const FloatType &) const = default;

private:
  details::FloatType m_type;
};

using f32 = float;
static_assert(sizeof(f32) == 4);
;
using f64 = double;
static_assert(sizeof(f64) == 8);

struct f16_reference;
struct f16_const_reference;
struct f16 {
  using reference = f16_reference;
  using const_reference = f16_const_reference;
  f16() : m_bits() {}
  explicit f16(float v) : m_bits(from_float(v)) {}
  explicit f16(double v) : m_bits(from_float(static_cast<float>(v))) {}

  explicit operator float() const { return to_float(m_bits); }
  explicit operator double() const {
    return static_cast<double>(to_float(m_bits));
  }

private:
  static inline uint16_t from_float(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x007FFFFF;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;

    if (exponent <= 0) {
      return static_cast<uint16_t>(sign);
    } else if (exponent >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00); // Inf
    }

    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
  }

  static inline float to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
      // Subnormal or zero
      if (mantissa == 0) {
        uint32_t bits = sign;
        return std::bit_cast<float>(bits);
      }
      // Normalize subnormal
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FF;
    } else if (exponent == 31) {
      // Inf or NaN
      uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
      return std::bit_cast<float>(bits);
    }

    exponent = exponent - 15 + 127;
    uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits);
  }

  std::uint16_t m_bits;
};

struct f16_const_reference {
  friend struct f16_reference;
  explicit f16_const_reference(const f16 &v) : m_ptr(&v) {}
  explicit f16_const_reference(const f16 *v) : m_ptr(v) {}

  explicit operator float() const { return static_cast<float>(*m_ptr); }
  explicit operator double() const { return static_cast<double>(*m_ptr); }
  explicit operator f16() const { return *m_ptr; }

private:
  const f16 *m_ptr;
};

struct f16_reference {
  explicit f16_reference(f16 &v) : m_ptr(&v) {}
  explicit f16_reference(f16 *v) : m_ptr(v) {}

  f16_reference &operator=(f16_reference v) {
    *m_ptr = *v.m_ptr;
    return *this;
  }

  f16_reference &operator=(f16_const_reference v) {
    *m_ptr = *v.m_ptr;
    return *this;
  }

  f16_reference &operator=(f16 v) {
    *m_ptr = v;
    return *this;
  }

  f16_reference &operator=(float v) {
    *m_ptr = f16{v};
    return *this;
  }

  f16_reference &operator=(double v) {
    *m_ptr = f16{v};
    return *this;
  }

  explicit operator float() const { return static_cast<float>(*m_ptr); }
  explicit operator double() const { return static_cast<double>(*m_ptr); }
  explicit operator f16() const { return *m_ptr; }

private:
  f16 *m_ptr;
};

// Reference to any float type (i.e. f16, f32, f64)
// includes runtime information about the underlying type.
struct fxx_const_reference;
struct fxx_reference {
  friend struct fxx_const_reference;
  explicit fxx_reference(f16 *v) : m_ptr(v), m_type(FloatType::F16) {}
  fxx_reference(f16 &v) : m_ptr(&v), m_type(FloatType::F16) {}

  explicit fxx_reference(f32 *v) : m_ptr(v), m_type(FloatType::F32) {}
  fxx_reference(f32 &v) : m_ptr(&v), m_type(FloatType::F32) {}

  explicit fxx_reference(f64 *v) : m_ptr(v), m_type(FloatType::F64) {}
  fxx_reference(f64 &v) : m_ptr(&v), m_type(FloatType::F64) {}

  explicit fxx_reference(void *ptr, FloatType type)
      : m_ptr(ptr), m_type(type) {}

  fxx_reference &operator=(f16 v) {
    if (m_type == FloatType::F16) {
      *reinterpret_cast<f16 *>(m_ptr) = v;
    } else if (m_type == FloatType::F32) {
      *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    } else if (m_type == FloatType::F64) {
      *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    } else {
      std::unreachable();
    }
    return *this;
  }

  fxx_reference &operator=(f16_reference v) {
    if (m_type == FloatType::F16) {
      *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
    } else if (m_type == FloatType::F32) {
      *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    } else if (m_type == FloatType::F64) {
      *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    } else {
      std::unreachable();
    }
    return *this;
  }

  fxx_reference &operator=(f16_const_reference v) {
    if (m_type == FloatType::F16) {
      *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
    } else if (m_type == FloatType::F32) {
      *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    } else if (m_type == FloatType::F64) {
      *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    } else {
      std::unreachable();
    }
    return *this;
  }

  fxx_reference &operator=(f32 v) {
    if (m_type == FloatType::F16) {
      *reinterpret_cast<f16 *>(m_ptr) = f16(v);
    } else if (m_type == FloatType::F32) {
      *reinterpret_cast<f32 *>(m_ptr) = v;
    } else if (m_type == FloatType::F64) {
      *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    } else {
      std::unreachable();
    }
    return *this;
  }

  fxx_reference &operator=(f64 v) {
    if (m_type == FloatType::F16) {
      *reinterpret_cast<f16 *>(m_ptr) = f16(v);
    } else if (m_type == FloatType::F32) {
      *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    } else if (m_type == FloatType::F64) {
      *reinterpret_cast<f64 *>(m_ptr) = v;
    } else {
      std::unreachable();
    }
    return *this;
  }

  fxx_reference &operator=(fxx_const_reference v);

  explicit operator f32() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return static_cast<f32>(v);
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return v;
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return static_cast<f32>(v);
    }
    std::unreachable();
  }
  explicit operator double() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return static_cast<f64>(v);
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return static_cast<f64>(v);
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return v;
    }
    std::unreachable();
  }
  explicit operator f16() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return v;
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return f16(v);
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return f16(v);
    }
    std::unreachable();
  }

private:
  void *m_ptr;
  FloatType m_type;
};

// Const Reference to any float type (i.e. f16, f32, f64)
// includes runtime information about the underlying type.
struct fxx_const_reference {
  friend struct fxx_reference;
  fxx_const_reference(fxx_reference ref)
      : m_ptr(ref.m_ptr), m_type(ref.m_type) {}

  explicit fxx_const_reference(const f16 *v)
      : m_ptr(v), m_type(FloatType::F16) {}

  fxx_const_reference(const f16 &v) : m_ptr(&v), m_type(FloatType::F16) {}

  explicit fxx_const_reference(const f32 *v)
      : m_ptr(v), m_type(FloatType::F32) {}
  fxx_const_reference(const f32 &v) : m_ptr(&v), m_type(FloatType::F32) {}

  explicit fxx_const_reference(const f64 *v)
      : m_ptr(v), m_type(FloatType::F64) {}
  fxx_const_reference(const f64 &v) : m_ptr(&v), m_type(FloatType::F64) {}

  explicit fxx_const_reference(const void *ptr, FloatType type)
      : m_ptr(ptr), m_type(type) {}

  explicit operator f32() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return static_cast<f32>(v);
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return v;
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return static_cast<f32>(v);
    }
    std::unreachable();
  }
  explicit operator double() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return static_cast<f64>(v);
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return static_cast<f64>(v);
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return v;
    }
    std::unreachable();
  }
  explicit operator f16() const {
    if (m_type == FloatType::F16) {
      f16 v = *reinterpret_cast<const f16 *>(m_ptr);
      return v;
    } else if (m_type == FloatType::F32) {
      f32 v = *reinterpret_cast<const f32 *>(m_ptr);
      return f16(v);
    } else if (m_type == FloatType::F64) {
      f64 v = *reinterpret_cast<const f64 *>(m_ptr);
      return f16(v);
    }
    std::unreachable();
  }

private:
  const void *m_ptr;
  FloatType m_type;
};

static_assert(sizeof(f16) == 2);

} // namespace vkcnn
