#include "./ActivationHostTensor.hpp"

namespace vkcnn {
ActivationHostTensorView::ActivationHostTensorView(ActivationHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

ActivationHostTensorView::ActivationHostTensorView(ActivationHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

ActivationHostTensorConstView::ActivationHostTensorConstView(
    const ActivationHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

ActivationHostTensorConstView::ActivationHostTensorConstView(
    const ActivationHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

fxx_reference ActivationHostTensorView::at(unsigned int w, unsigned int h,
                                           unsigned int c) {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  std::byte *ptr = m_buffer + offset;
  return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
}

fxx_const_reference ActivationHostTensorView::at(unsigned int w, unsigned int h,
                                                 unsigned int c) const {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return fxx_const_reference(reinterpret_cast<const void *>(ptr), m_desc.type);
}

fxx_reference ActivationHostTensorView::operator[](unsigned int linearIndex) {
  std::size_t offset = linearIndex * m_desc.type.size();
  std::byte *ptr = m_buffer + offset;
  return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
}

fxx_const_reference
ActivationHostTensorView::operator[](unsigned int linearIndex) const {
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return fxx_const_reference(reinterpret_cast<const void *>(ptr), m_desc.type);
}
fxx_const_reference ActivationHostTensorConstView::at(unsigned int w,
                                                      unsigned int h,
                                                      unsigned int c) const {
  std::size_t linearIndex = m_desc.layout(m_desc.shape, w, h, c);
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return fxx_const_reference(reinterpret_cast<const void *>(ptr), m_desc.type);
}
fxx_const_reference
ActivationHostTensorConstView::operator[](unsigned int linearIndex) const {
  std::size_t offset = linearIndex * m_desc.type.size();
  const std::byte *ptr = m_buffer + offset;
  return fxx_const_reference(reinterpret_cast<const void *>(ptr), m_desc.type);
}
} // namespace vkcnn
