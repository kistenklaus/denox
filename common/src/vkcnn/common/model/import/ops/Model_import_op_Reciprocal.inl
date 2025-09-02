#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Reciprocal(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> & /*attributes*/,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // ---- arity ----
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Reciprocal \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Reciprocal \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &X = *inputs[0];
  if (!X.isHost())
    throw std::runtime_error(
        fmt::format("vkcnn: Reciprocal \"{}\": only HostTensor is supported.",
                    node.name()));

  const HostTensor &Xin = X.host();

  // Require constant shape/view (we rely on constIndexOf)
  if (!Xin.isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Reciprocal \"{}\": input must have constant shape.",
                    node.name()));
  if (!Xin.view().isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Reciprocal \"{}\": input must have constant view.",
                    node.name()));

  const Dtype dt = Xin.type();
  const bool isF32 = (dt == Dtype::Float32);
  const bool isF64 = (dt == Dtype::Float64);

  if (!isF32 && !isF64)
    throw std::runtime_error(
        fmt::format("vkcnn: Reciprocal \"{}\": unsupported dtype {}. "
                    "Supported: Float32, Float64.",
                    node.name(), dtype_to_string(dt)));

  // ---- output shape / size ----
  const TensorShape outShape = Xin.shape();
  const auto outDims = outShape.toU64();
  std::size_t N = 1;
  for (auto d : outDims)
    N *= (std::size_t)d;

  auto next_indexer = [&](auto &idxVec) {
    for (std::size_t ax = idxVec.size(); ax-- > 0;) {
      if (++idxVec[ax] < outDims[ax])
        return true;
      idxVec[ax] = 0;
    }
    return false;
  };

  auto make_out =
      [&](Dtype odt,
          std::size_t elemSize) -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw)
      throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(odt, raw, N * elemSize));
  };

  if (isF32) {
    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float *>(outStore->data());
    const auto src = Xin.storage()->f32();

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const float v = src[lin];
      dst[i] = 1.0f / v; // IEEE-754: handles 0, +/-inf, NaN naturally
      if (!next_indexer(idx))
        break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // Float64
  {
    auto outStore = make_out(Dtype::Float64, sizeof(double));
    double *dst = reinterpret_cast<double *>(outStore->data());
    const auto src = Xin.storage()->f64();

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const double v = src[lin];
      dst[i] = 1.0 / v;
      if (!next_indexer(idx))
        break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }
}

} // namespace vkcnn::details
