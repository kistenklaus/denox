#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> ceil(
    [[maybe_unused]] ImportState &state,
    memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\" expects 1 input.", nodeName));
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\": input is required.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &xT = *inputs[0];

  // Host-only
  if (!xT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\": DeviceTensor is not supported.", nodeName));

  const HostTensor &Xin = xT.host();

  // Only floating types; keep dtype unchanged
  const Dtype dt = Xin.type();
  if (!dt.toDenoxType().has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\": only floating HostTensors are supported (got {}).",
        nodeName, dt.to_string()));

  // Work on a contiguous view for simple linear iteration
  HostTensor Xc = Xin.contiguous();

  // Shape must be known (so we can allocate exact size)
  if (!Xc.shape().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Ceil \"{}\": dynamic host shapes are unsupported.", nodeName));

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
  switch (dt.kind()) {
  case DtypeKind::Float32: {
    const float *src = reinterpret_cast<const float *>(srcRaw);
    float *dst = reinterpret_cast<float *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = std::ceil(src[i]);
    break;
  }
  case DtypeKind::Float64: {
    const double *src = reinterpret_cast<const double *>(srcRaw);
    double *dst = reinterpret_cast<double *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = std::ceil(src[i]);
    break;
  }
  case DtypeKind::Undefined:
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    std::free(raw);
    throw std::runtime_error(
        fmt::format("vkcnn: Ceil \"{}\": unsupported floating dtype {} "
                    "(supported: f32, f64).",
                    nodeName, dt.to_string()));
  }

  // Same shape; identity view
  TensorShape outShape = Xc.shape();
  HostTensor outHT(outShape, std::move(dstStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace denox::onnx::details::ops
