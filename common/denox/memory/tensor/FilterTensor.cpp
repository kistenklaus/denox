#include "denox/memory/tensor/FilterTensor.hpp"

namespace denox::memory {

FilterTensorView::FilterTensorView(FilterTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

FilterTensorView::FilterTensorView(FilterTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

FilterTensorConstView::FilterTensorConstView(
    const FilterTensor *tensor)
    : m_desc(tensor->desc()), m_buffer(tensor->data()) {}

FilterTensorConstView::FilterTensorConstView(
    const FilterTensor &tensor)
    : m_desc(tensor.desc()), m_buffer(tensor.data()) {}

void FilterTensorView::assignFrom(const FilterTensorConstView &view) {
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
