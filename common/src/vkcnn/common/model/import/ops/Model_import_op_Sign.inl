#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Sign(
    [[maybe_unused]] ImportState & state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Attribute> & attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  // ---- arity ----
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sign \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Sign \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &X = *inputs[0];

  // Host-only for now
  if (!X.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sign \"{}\": only HostTensor is supported.", node.name()));

  const HostTensor &H = X.host();

  // No symbolic dtype for sign (explicit request)
  if (H.type() == Dtype::Sym)
    throw std::runtime_error(
        fmt::format("vkcnn: Sign \"{}\": symbolic tensors are not supported.",
                    node.name()));
  if (!H.isConstant() || !H.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sign \"{}\": input must have constant shape and view.",
        node.name()));

  const TensorShape shape = H.shape();
  const auto dims = shape.toU64();
  std::size_t N = 1;
  for (auto d : dims)
    N *= (std::size_t)d;

  // Helpers for indexing
  const size_t r = dims.size();
  std::vector<std::uint64_t> idx(r, 0);
  auto step = [&]() -> bool {
    if (idx.empty())
      return false;
    for (size_t ax = r; ax-- > 0;) {
      if (++idx[ax] < dims[ax])
        return true;
      idx[ax] = 0;
    }
    return false;
  };

  const Dtype dt = H.type();

  // Float32
  if (dt == Dtype::Float32) {
    auto outRaw = std::malloc(N * sizeof(float));
    if (!outRaw)
      throw std::bad_alloc();
    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Float32, outRaw, N * sizeof(float)));
    float *dst = reinterpret_cast<float *>(outStore->data());
    const auto src = H.storage()->f32();

    if (r == 0) {
      const std::size_t si = H.view().constIndexOf({});
      const float x = src[si];
      dst[0] = (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
    } else {
      for (std::size_t i = 0;; ++i) {
        const std::size_t si = H.view().constIndexOf({idx.data(), idx.size()});
        const float x = src[si];
        dst[i] = (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
        if (!step())
          break;
      }
    }

    HostTensor out(shape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // Float64
  if (dt == Dtype::Float64) {
    auto outRaw = std::malloc(N * sizeof(double));
    if (!outRaw)
      throw std::bad_alloc();
    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Float64, outRaw, N * sizeof(double)));
    double *dst = reinterpret_cast<double *>(outStore->data());
    const auto src = H.storage()->f64();

    if (r == 0) {
      const std::size_t si = H.view().constIndexOf({});
      const double x = src[si];
      dst[0] = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    } else {
      for (std::size_t i = 0;; ++i) {
        const std::size_t si = H.view().constIndexOf({idx.data(), idx.size()});
        const double x = src[si];
        dst[i] = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
        if (!step())
          break;
      }
    }

    HostTensor out(shape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // Int64
  if (dt == Dtype::Int64) {
    auto outRaw = std::malloc(N * sizeof(std::int64_t));
    if (!outRaw)
      throw std::bad_alloc();
    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Int64, outRaw, N * sizeof(std::int64_t)));
    std::int64_t *dst = reinterpret_cast<std::int64_t *>(outStore->data());
    const auto src = H.storage()->i64();

    if (r == 0) {
      const std::size_t si = H.view().constIndexOf({});
      const std::int64_t x = src[si];
      dst[0] = (x > 0) ? 1 : ((x < 0) ? -1 : 0);
    } else {
      for (std::size_t i = 0;; ++i) {
        const std::size_t si = H.view().constIndexOf({idx.data(), idx.size()});
        const std::int64_t x = src[si];
        dst[i] = (x > 0) ? 1 : ((x < 0) ? -1 : 0);
        if (!step())
          break;
      }
    }

    HostTensor out(shape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // If you have additional integral types supported by HostTensorStorage
  // (e.g., Int32), add similar branches here. For now, conservatively
  // reject other dtypes (strings, bools, etc.).
  throw std::runtime_error(
      fmt::format("vkcnn: Sign \"{}\": unsupported dtype {} (non-symbolic).",
                  node.name(), dtype_to_string(dt)));
}

} // namespace vkcnn::details
