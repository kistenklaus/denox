#include "frontend/onnx/details/values/Tensor.hpp"

namespace denox::onnx::details {

Tensor Tensor::Host(HostTensor hostTensor) {
  return Tensor{std::make_shared<HostTensor>(std::move(hostTensor))};
}

Tensor Tensor::Device(DeviceTensor deviceTensor) {
  return Tensor{std::make_shared<DeviceTensor>(std::move(deviceTensor))};
}

Tensor Tensor::Host(memory::shared_ptr<HostTensor> hostTensor) {
  return Tensor{std::move(hostTensor)};
}

Tensor Tensor::Device(memory::shared_ptr<DeviceTensor> deviceTensor) {
  return Tensor{std::move(deviceTensor)};
}

bool Tensor::isDevice() const {
  return std::holds_alternative<memory::shared_ptr<DeviceTensor>>(m_rep);
}

bool Tensor::isHost() const {
  return std::holds_alternative<memory::shared_ptr<HostTensor>>(m_rep);
}

const DeviceTensor &Tensor::device() const {
  assert(isDevice());
  return *std::get<memory::shared_ptr<DeviceTensor>>(m_rep);
}

const HostTensor &Tensor::host() const {
  assert(isHost());
  return *std::get<memory::shared_ptr<HostTensor>>(m_rep);
}

TensorShape Tensor::shape() const {
  if (isDevice()) {
    return device().shape();
  } else if (isHost()) {
    return host().shape();
  } else {
    throw std::logic_error("unreachable");
  }
}

std::size_t Tensor::rank() const {
  if (isDevice()) {
    return device().rank();
  } else if (isHost()) {
    return host().rank();
  } else {
    throw std::logic_error("unreachable");
  }
}

memory::optional<Dtype> Tensor::type() {
  if (isDevice()) {
    return device().type();
  } else if (isHost()) {
    return host().type();
  } else {
    throw std::logic_error("unreachable");
  }
}

} // namespace denox::onnx::details
