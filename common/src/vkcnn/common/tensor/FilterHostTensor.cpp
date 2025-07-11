#include "./FilterHostTensor.hpp"

namespace vkcnn {

FilterHostTensorView::FilterHostTensorView(FilterHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

FilterHostTensorView::FilterHostTensorView(FilterHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

FilterHostTensorConstView::FilterHostTensorConstView(
    const FilterHostTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

FilterHostTensorConstView::FilterHostTensorConstView(
    const FilterHostTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

void FilterHostTensorView::assignFrom(const FilterHostTensorConstView &view) {
  for (unsigned int r = 0; r < shape().r; ++r) {
    for (unsigned int s = 0; s < shape().s; ++s) {
      for (unsigned int k = 0; k < shape().k; ++k) {
        for (unsigned int c = 0; c < shape().c; ++c) {
          this->at(s, r, c, k) = view.at(s, r, c, k);
        }
      }
    }
  }
}
} // namespace vkcnn
