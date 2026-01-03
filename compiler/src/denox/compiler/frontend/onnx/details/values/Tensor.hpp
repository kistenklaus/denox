#pragma once

#include "denox/compiler/frontend/onnx/details/values/DeviceTensor.hpp"
#include "denox/compiler/frontend/onnx/details/values/HostTensor.hpp"
#include "denox/memory/container/shared_ptr.hpp"
#include "denox/memory/container/variant.hpp"

namespace onnx {
class TensorProto;
}

namespace denox::onnx::details {

class Tensor {
  using Rep = memory::variant<memory::shared_ptr<HostTensor>,
                              memory::shared_ptr<DeviceTensor>>;

public:
  static Tensor Host(HostTensor hostTensor);
  static Tensor Device(DeviceTensor deviceTensor);
  static Tensor Host(memory::shared_ptr<HostTensor> hostTensor);
  static Tensor Device(memory::shared_ptr<DeviceTensor> deviceTensor);
  bool isDevice() const;
  bool isHost() const;
  const DeviceTensor &device() const;
  DeviceTensor &device();
  const HostTensor &host() const;
  TensorShape shape() const;
  std::size_t rank() const;
  memory::optional<Dtype> type();

private:
  Tensor(Rep rep) : m_rep(std::move(rep)) {}
  Rep m_rep;
};

} // namespace denox::onnx::details
