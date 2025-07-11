#include "./FloatType.hpp"

namespace vkcnn {

fxx_reference &fxx_reference::operator=(fxx_const_reference v) {
  if (m_type == FloatType::F16) {
    *reinterpret_cast<f16 *>(m_ptr) = static_cast<f16>(v);
    return *this;
  } else if (m_type == FloatType::F32) {
    *reinterpret_cast<f32 *>(m_ptr) = static_cast<f32>(v);
    return *this;
  } else if (m_type == FloatType::F64) {
    *reinterpret_cast<f64 *>(m_ptr) = static_cast<f64>(v);
    return *this;
  }
  std::unreachable();
}
} // namespace vkcnn
