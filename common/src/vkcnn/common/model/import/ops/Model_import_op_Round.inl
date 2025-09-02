#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Round(
    ImportState & /*state*/, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> & /*attributes*/,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {
  // ---- arity ----
  if (inputs.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\" expects 1 input.", node.name()));
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\": input is required.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Round \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &xT = *inputs[0];

  // Host-only
  if (!xT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Round \"{}\": DeviceTensor is not supported.", node.name()));

  const HostTensor &Xin = xT.host();

  // Only floating types; keep dtype unchanged
  const Dtype dt = Xin.type();
  if (!dtype_to_float_type(dt).has_value())
    throw std::runtime_error(fmt::format("vkcnn: Round \"{}\": only floating "
                                         "HostTensors are supported (got {}).",
                                         node.name(), dtype_to_string(dt)));

  // Work on a contiguous view for simple linear iteration
  HostTensor Xc = Xin.contiguous();

  // Shape must be known (so we can allocate exact size)
  if (!Xc.shape().isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\": dynamic host shapes are unsupported.",
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
  switch (dt) {
  case Dtype::Float32: {
    const float *src = reinterpret_cast<const float *>(srcRaw);
    float *dst = reinterpret_cast<float *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = round_even_f32(src[i]);
    break;
  }
  case Dtype::Float64: {
    const double *src = reinterpret_cast<const double *>(srcRaw);
    double *dst = reinterpret_cast<double *>(raw);
    for (std::size_t i = 0; i < count; ++i)
      dst[i] = round_even_f64(src[i]);
    break;
  }
  default:
    std::free(raw);
    throw std::runtime_error(
        fmt::format("vkcnn: Round \"{}\": unsupported floating dtype {} "
                    "(supported: f32, f64).",
                    node.name(), dtype_to_string(dt)));
  }

  // Same shape; identity view
  TensorShape outShape = Xc.shape();
  HostTensor outHT(outShape, std::move(dstStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace vkcnn::details
