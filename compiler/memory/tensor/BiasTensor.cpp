#include "./BiasTensor.hpp"

namespace denox::memory {

BiasTensorView::BiasTensorView(BiasTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasTensorView::BiasTensorView(BiasTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

void BiasTensorView::assignFrom(const BiasTensorConstView &view) {
  assert(shape() == view.shape());
  for (unsigned int c = 0; c < view.shape(); ++c) {
    this->at(c) = view.at(c);
  }
}

BiasTensorConstView::BiasTensorConstView(const BiasTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasTensorConstView::BiasTensorConstView(const BiasTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}
} // namespace vkcnn
