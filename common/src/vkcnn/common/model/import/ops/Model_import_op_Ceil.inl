#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Ceil(
    [[maybe_unused]] ImportState &state,
    std::span<const std::optional<Tensor>> inputs, std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {
  // ---- arity ----
  if (inputs.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\" expects 1 input.", node.name()));
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\": input is required.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &xT = *inputs[0];

  // Host-only
  if (!xT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\": DeviceTensor is not supported.", node.name()));

  const HostTensor &Xin = xT.host();

  // Only floating types; keep dtype unchanged
  const Dtype dt = Xin.type();
  if (!dtype_to_float_type(dt).has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\": only floating HostTensors are supported (got {}).",
        node.name(), dtype_to_string(dt)));

  // Work on a contiguous view for simple linear iteration
  HostTensor Xc = Xin.contiguous();

  // Shape must be known (so we can allocate exact size)
  if (!Xc.shape().isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\": dynamic host shapes are unsupported.",
                    node.name()));

  const auto dimsU64 = Xc.shape().toU64();
  std::size_t count = 1;
  for (auto d : dimsU64)
    count *= static_cast<std::size_t>(d);

  const std::size_t elem = Xc.elemSize();

  // Allocate destination storage of same dtype/size
  void *raw = std::malloc(count * elem);
  if (!raw)
    throw std::bad_alloc();
  auto dstStore = std::make_shared<HostTensorStorage>(
      HostTensorStorage::TakeOwnership(dt, raw, count * elem));

  // Elementwise ceil, dtype preserved
  const void *srcRaw = Xc.storage()->data();
  switch (dt) {
  case Dtype::Float32: {
    const float *src = reinterpret_cast<const float *>(srcRaw);
    float *dst = reinterpret_cast<float *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = std::ceil(src[i]);
    break;
  }
  case Dtype::Float64: {
    const double *src = reinterpret_cast<const double *>(srcRaw);
    double *dst = reinterpret_cast<double *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = std::ceil(src[i]);
    break;
  }
  default:
    std::free(raw);
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\": unsupported floating dtype {} "
                    "(supported: f32, f64).",
                    node.name(), dtype_to_string(dt)));
  }

  // Same shape; identity view
  TensorShape outShape = Xc.shape();
  HostTensor outHT(outShape, std::move(dstStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace vkcnn::details
