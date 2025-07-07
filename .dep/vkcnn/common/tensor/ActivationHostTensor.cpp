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

} // namespace vkcnn
