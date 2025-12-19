#pragma once

#include <cstdint>
namespace denox::memory {

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
  static std::uint16_t from_float(float f);

  static float to_float(uint16_t h);

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


} // namespace denox::compiler
