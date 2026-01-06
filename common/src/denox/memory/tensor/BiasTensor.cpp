#include "denox/memory/tensor/BiasTensor.hpp"

namespace denox::memory {
BiasTensorView::BiasTensorView(BiasTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasTensorView::BiasTensorView(BiasTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

void BiasTensorView::assignFrom(const BiasTensorConstView &view) {
  assert(shape() == view.shape());
  if (m_desc.layout.isVectorized()) {
    std::memset(m_buffer, 0, byteSize());
  }
  for (unsigned int c = 0; c < view.shape(); ++c) {
    auto v = view.at(c);
    this->at(c) = v;
  }
}

BiasTensorConstView::BiasTensorConstView(const BiasTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasTensorConstView::BiasTensorConstView(const BiasTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}
} // namespace denox::memory
