#include "denox/memory/tensor/ActivationTensor.hpp"

namespace denox::memory {

ActivationTensorView::ActivationTensorView(ActivationTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

ActivationTensorView::ActivationTensorView(ActivationTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

ActivationTensorConstView::ActivationTensorConstView(
    const ActivationTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

ActivationTensorConstView::ActivationTensorConstView(
    const ActivationTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

dtype_reference ActivationTensorView::at(unsigned int w, unsigned int h,
                                         unsigned int c) {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  std::byte *ptr = m_buffer + offset;
  return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
}

dtype_const_reference ActivationTensorView::at(unsigned int w, unsigned int h,
                                               unsigned int c) const {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
}

dtype_reference ActivationTensorView::operator[](unsigned int linearIndex) {
  std::size_t offset = linearIndex * m_desc.type.size();
  std::byte *ptr = m_buffer + offset;
  return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
}

dtype_const_reference
ActivationTensorView::operator[](unsigned int linearIndex) const {
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
}

dtype_const_reference ActivationTensorConstView::at(unsigned int w,
                                                    unsigned int h,
                                                    unsigned int c) const {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
}

dtype_const_reference
ActivationTensorConstView::operator[](unsigned int linearIndex) const {
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
}

} // namespace denox::compiler
