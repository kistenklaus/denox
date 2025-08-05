#include "./BiasHostTensor.hpp"

namespace vkcnn {

BiasHostTensorView::BiasHostTensorView(BiasHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasHostTensorView::BiasHostTensorView(BiasHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

void BiasHostTensorView::assignFrom(const BiasHostTensorConstView &view) {
  assert(shape() == view.shape());
  for (unsigned int c = 0; c < view.shape(); ++c) {
    this->at(c) = view.at(c);
  }
}

BiasHostTensorConstView::BiasHostTensorConstView(const BiasHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

BiasHostTensorConstView::BiasHostTensorConstView(const BiasHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}
} // namespace vkcnn
