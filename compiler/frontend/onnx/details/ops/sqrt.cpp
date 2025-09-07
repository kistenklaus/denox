#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
sqrt([[maybe_unused]] ImportState &state,
     memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
         &attributes,
     [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // ---- arity ----
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Sqrt \"{}\" expects exactly 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Sqrt \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &X = *inputs[0];
  if (!X.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sqrt \"{}\": only HostTensor is supported.", nodeName));

  const HostTensor &Xin = X.host();

  // Require constant shape/view (we rely on constIndexOf)
  if (!Xin.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sqrt \"{}\": input must have constant shape.", nodeName));
  if (!Xin.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Sqrt \"{}\": input must have constant view.", nodeName));

  const Dtype dt = Xin.type();
  const bool isF32 = (dt == Dtype::Float32);
  const bool isF64 = (dt == Dtype::Float64);

  if (!isF32 && !isF64)
    throw std::runtime_error(
        fmt::format("vkcnn: Sqrt \"{}\": unsupported dtype {}. "
                    "Supported: Float32, Float64.",
                    nodeName, dt.to_string()));

  // ---- output shape / size ----
  const TensorShape outShape = Xin.shape();
  const auto outDims = outShape.toU64();
  std::size_t N = 1;
  for (auto d : outDims)
    N *= static_cast<std::size_t>(d);

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

    memory::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const float v = src[lin];
      dst[i] = std::sqrt(v); // IEEE-754: negative → NaN
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

    memory::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const double v = src[lin];
      dst[i] = std::sqrt(v);
      if (!next_indexer(idx))
        break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }
}

} // namespace denox::onnx::details::ops
