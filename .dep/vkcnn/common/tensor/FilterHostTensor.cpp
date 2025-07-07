#include "./FilterHostTensor.hpp"
#include <stdexcept>

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
  auto assignWith = [this, view](auto tag) {
    using T = decltype(tag);
    for (unsigned int r = 0; r < shape().r; ++r) {
      for (unsigned int s = 0; s < shape().s; ++s) {
        for (unsigned int k = 0; k < shape().k; ++k) {
          for (unsigned int c = 0; c < shape().c; ++c) {
            this->at<T>(s, r, c, k) = view.at<T>(s, r, c, k);
          }
        }
      }
    }
  };
  auto type = floatType();
  if (type == FloatType::F16) {
    assignWith(f16{});
  } else if (type == FloatType::F32) {
    assignWith(f32{});
  } else if (type == FloatType::F64) {
    assignWith(f64{});
  } else {
    throw std::runtime_error("Invalid FloatType");
  }
}
} // namespace vkcnn
