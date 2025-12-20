#pragma once

#include "denox/compiler/frontend/onnx/details/values/HostTensorStorage.hpp"
#include "denox/compiler/frontend/onnx/details/values/TensorShape.hpp"
#include "denox/compiler/frontend/onnx/details/values/TensorViewDesc.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/shared_ptr.hpp"
#include "denox/memory/container/span.hpp"

namespace onnx {
class TensorProto;
}

namespace denox::onnx::details {

class HostTensor {
public:
  explicit HostTensor(TensorShape shape,
                      memory::shared_ptr<HostTensorStorage> storage);

  TensorShape shape() const { return m_shape; }
  TensorViewDesc view() const { return m_view; }
  Dtype type() const { return m_store->type(); }
  const memory::shared_ptr<HostTensorStorage> &storage() const;
  size_t rank() const { return m_shape.rank(); }
  bool isConstant() const { return m_shape.isConstant(); }
  compiler::Symbolic numel() const { return m_shape.numel(); }
  std::size_t elemSize() const { return type().size(); }
  bool isContiguous() const;
  std::size_t sizeBytesIfStatic() const;
  std::size_t byteOffset() const;
  const void *data() const;
  void *data();
  HostTensor withView(TensorShape newShape, TensorViewDesc newView) const;
  HostTensor permute(memory::span<const int64_t> perm) const;
  HostTensor unsqueeze(std::size_t axis) const;
  HostTensor squeeze(std::size_t axis) const;
  HostTensor materializeContiguous() const;
  std::size_t sizeElemsIfStatic() const;
  HostTensor contiguous() const;
  HostTensor reshape(const TensorShape &newShape) const;
  HostTensor select(std::size_t axis, std::uint64_t index) const;
  HostTensor narrow(std::size_t axis, std::uint64_t start,
                    std::uint64_t length) const;
  HostTensor broadcastInDim(const TensorShape &to,
                            memory::span<const int64_t> axesMap) const;
  HostTensor clone() const;
  bool sameStorageAs(const HostTensor &o) const;

  template <class T>
  [[deprecated("Use explicit interfaces instead, no need for templates")]]
  memory::span<const T> span() const {
    assert(isContiguous());
    assert(sizeof(T) == elemSize());
    const auto *p = static_cast<const T *>(data());
    // count:
    std::size_t n = 1;
    for (auto d : m_shape.toU64())
      n *= static_cast<std::size_t>(d);
    return {p, n};
  }

  memory::span<const float> floats() const;

  /// loads a index as a i64, possibly upcasts!
  std::int64_t loadI64(memory::span<const std::uint64_t> idx) const;
  std::uint64_t loadU64(memory::span<const std::uint64_t> idx) const;
  double loadDouble(memory::span<const std::uint64_t> idx) const;
  Sym loadSym(memory::span<const std::uint64_t> idx) const;

  static HostTensor parse(const ::onnx::TensorProto &tensor,
                          const io::Path &externalDir = {});

private:
  explicit HostTensor(TensorShape shape, TensorViewDesc view,
                      memory::shared_ptr<HostTensorStorage> storage);

private:
  TensorShape m_shape;
  TensorViewDesc m_view;
  memory::shared_ptr<HostTensorStorage> m_store;
};

} // namespace denox::onnx::details
