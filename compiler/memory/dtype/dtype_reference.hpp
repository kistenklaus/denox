#pragma once

#include "memory/dtype/dtype.hpp"
#include "memory/dtype/f16.hpp"
#include "memory/dtype/f32.hpp"
#include "memory/dtype/f64.hpp"

namespace denox::memory {

struct dtype_const_reference;

struct dtype_reference {
  friend struct dtype_const_reference;
  explicit dtype_reference(f16 *v) : m_ptr(v), m_type(Dtype::F16) {}
  dtype_reference(f16 &v) : m_ptr(&v), m_type(Dtype::F16) {}

  explicit dtype_reference(f32 *v) : m_ptr(v), m_type(Dtype::F32) {}
  dtype_reference(f32 &v) : m_ptr(&v), m_type(Dtype::F32) {}

  explicit dtype_reference(f64 *v) : m_ptr(v), m_type(Dtype::F64) {}
  dtype_reference(f64 &v) : m_ptr(&v), m_type(Dtype::F64) {}

  explicit dtype_reference(void *ptr, Dtype type) : m_ptr(ptr), m_type(type) {}

  dtype_reference &operator=(f16 v);

  dtype_reference &operator=(dtype_reference v);

  dtype_reference &operator=(f16_const_reference v);

  dtype_reference &operator=(f32 v);

  dtype_reference &operator=(f64 v);

  dtype_reference &operator=(dtype_const_reference v);

  explicit operator f32() const;
  explicit operator double() const;
  explicit operator f16() const;

private:
  void *m_ptr;
  Dtype m_type;
};

struct dtype_const_reference {
  friend struct fxx_reference;
  dtype_const_reference(dtype_reference ref)
      : m_ptr(ref.m_ptr), m_type(ref.m_type) {}

  explicit dtype_const_reference(const f16 *v) : m_ptr(v), m_type(Dtype::F16) {}

  dtype_const_reference(const f16 &v) : m_ptr(&v), m_type(Dtype::F16) {}

  explicit dtype_const_reference(const f32 *v) : m_ptr(v), m_type(Dtype::F32) {}
  dtype_const_reference(const f32 &v) : m_ptr(&v), m_type(Dtype::F32) {}

  explicit dtype_const_reference(const f64 *v) : m_ptr(v), m_type(Dtype::F64) {}
  dtype_const_reference(const f64 &v) : m_ptr(&v), m_type(Dtype::F64) {}

  explicit dtype_const_reference(const void *ptr, Dtype type)
      : m_ptr(ptr), m_type(type) {}

  explicit operator f32() const;
  explicit operator double() const;
  explicit operator f16() const;

private:
  const void *m_ptr;
  Dtype m_type;
};

} // namespace denox::compiler
