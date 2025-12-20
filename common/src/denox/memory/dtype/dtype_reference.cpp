#include "denox/memory/dtype/dtype_reference.hpp"
#include "denox/diag/unreachable.hpp"

namespace denox::memory {

dtype_reference &dtype_reference::operator=(f16 v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = v;
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
  } else {
    diag::unreachable();
  }
  return *this;
}

dtype_reference &dtype_reference::operator=(dtype_reference v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
  } else {
    diag::unreachable();
  }
  return *this;
}

dtype_reference &dtype_reference::operator=(f16_const_reference v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
  } else {
    diag::unreachable();
  }
  return *this;
}

dtype_reference &dtype_reference::operator=(f32 v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = f16(v);
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = v;
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
  } else {
    diag::unreachable();
  }
  return *this;
}

dtype_reference &dtype_reference::operator=(f64 v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = f16(v);
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = v;
  } else {
    diag::unreachable();
  }
  return *this;
}

dtype_reference &dtype_reference::operator=(dtype_const_reference v) {
  if (m_type == Dtype::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
    return *this;
  } else if (m_type == Dtype::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    return *this;
  } else if (m_type == Dtype::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    return *this;
  }
  diag::unreachable();
}

dtype_reference::operator f32() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return static_cast<f32>(v);
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return v;
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return static_cast<f32>(v);
  }
  diag::unreachable();
}

dtype_reference::operator double() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return static_cast<f64>(v);
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return static_cast<f64>(v);
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return v;
  }
  diag::unreachable();
}

dtype_reference::operator f16() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return v;
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return f16(v);
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return f16(v);
  }
  diag::unreachable();
}

dtype_const_reference::operator f32() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return static_cast<f32>(v);
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return v;
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return static_cast<f32>(v);
  }
  diag::unreachable();
}

dtype_const_reference::operator double() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return static_cast<f64>(v);
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return static_cast<f64>(v);
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return v;
  }
  diag::unreachable();
}

dtype_const_reference::operator f16() const {
  if (m_type == Dtype::F16) {
    f16 v = *reinterpret_cast<const f16 *>(m_ptr);
    return v;
  } else if (m_type == Dtype::F32) {
    f32 v = *reinterpret_cast<const f32 *>(m_ptr);
    return f16(v);
  } else if (m_type == Dtype::F64) {
    f64 v = *reinterpret_cast<const f64 *>(m_ptr);
    return f16(v);
  }
  diag::unreachable();
}

} // namespace denox::compiler
