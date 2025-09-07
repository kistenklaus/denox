#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
round([[maybe_unused]] ImportState &state,
      memory::span<const memory::optional<Tensor>> inputs,
      std::size_t outputCount,
      [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
          &attributes,
      [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\" expects 1 input.", nodeName));
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\": input is required.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Round \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &xT = *inputs[0];

  // Host-only
  if (!xT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Round \"{}\": DeviceTensor is not supported.", nodeName));

  const HostTensor &Xin = xT.host();

  // Only floating types; keep dtype unchanged
  const Dtype dt = Xin.type();
  if (!dt.toDenoxType().has_value())
    throw std::runtime_error(fmt::format("vkcnn: Round \"{}\": only floating "
                                         "HostTensors are supported (got {}).",
                                         nodeName, dt.to_string()));

  // Work on a contiguous view for simple linear iteration
  HostTensor Xc = Xin.contiguous();

  // Shape must be known (so we can allocate exact size)
  if (!Xc.shape().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Round \"{}\": dynamic host shapes are unsupported.", nodeName));

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

  // Tie-to-even helper
  auto round_even_f32 = [](float x) -> float {
    if (std::isnan(x) || std::isinf(x))
      return x;
    float f = std::floor(x);
    float d = x - f; // in [0,1)
    if (d < 0.5f)
      return f;
    if (d > 0.5f)
      return f + 1.0f;
    // tie: pick even
    return (std::fmod(f, 2.0f) != 0.0f) ? (f + 1.0f) : f;
  };
  auto round_even_f64 = [](double x) -> double {
    if (std::isnan(x) || std::isinf(x))
      return x;
    double f = std::floor(x);
    double d = x - f; // in [0,1)
    if (d < 0.5)
      return f;
    if (d > 0.5)
      return f + 1.0;
    // tie: pick even
    return (std::fmod(f, 2.0) != 0.0) ? (f + 1.0) : f;
  };

  // Elementwise round-to-even, dtype preserved
  const void *srcRaw = Xc.storage()->data();
  switch (dt.kind()) {
  case DtypeKind::Float32: {
    const float *src = static_cast<const float *>(srcRaw);
    float *dst = static_cast<float *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = round_even_f32(src[i]);
    break;
  }
  case DtypeKind::Float64: {
    const double *src = static_cast<const double *>(srcRaw);
    double *dst = static_cast<double *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = round_even_f64(src[i]);
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
        fmt::format("vkcnn: Round \"{}\": unsupported floating dtype {} "
                    "(supported: f32, f64).",
                    nodeName, dt.to_string()));
  }

  // Same shape; identity view
  TensorShape outShape = Xc.shape();
  HostTensor outHT(outShape, std::move(dstStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace denox::onnx::details::ops
