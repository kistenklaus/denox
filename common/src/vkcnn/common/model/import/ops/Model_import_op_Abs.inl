#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Abs(ImportState &state, std::span<const std::optional<Tensor>> inputs,
              std::size_t outputCount,
              [[maybe_unused]] const std::unordered_map<std::string, Attribute> & attributes,
              [[maybe_unused]] opset_version version,
              const onnx::NodeProto &node) {

  // ---- arity ----
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Abs \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Abs \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &X = *inputs[0];
  if (!X.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Abs \"{}\": only HostTensor is supported.", node.name()));

  const HostTensor &Xin = X.host();

  // Require constant shape/view (we rely on constIndexOf)
  if (!Xin.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Abs \"{}\": input must have constant shape.", node.name()));
  if (!Xin.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Abs \"{}\": input must have constant view.", node.name()));

  const Dtype dt = Xin.type();
  const bool isF32 = (dt == Dtype::Float32);
  const bool isF64 = (dt == Dtype::Float64);
  const bool isI32 = (dt == Dtype::Int32);
  const bool isI64 = (dt == Dtype::Int64);
  const bool isSym = (dt == Dtype::Sym);

  if (!isF32 && !isF64 && !isI32 && !isI64 && !isSym)
    throw std::runtime_error(
        fmt::format("vkcnn: Abs \"{}\": unsupported dtype {}. "
                    "Supported: Float32, Float64, Int32, Int64, Sym.",
                    node.name(), dtype_to_string(dt)));

  // ---- output shape / size ----
  const TensorShape outShape = Xin.shape();
  const auto outDims = outShape.toU64();
  std::size_t N = 1;
  for (auto d : outDims)
    N *= (std::size_t)d;

  // ---- index iterator ----
  auto next_indexer = [&](auto &idxVec) {
    for (std::size_t ax = idxVec.size(); ax-- > 0;) {
      if (++idxVec[ax] < outDims[ax])
        return true;
      idxVec[ax] = 0;
    }
    return false;
  };

  // ---- helpers ----
  auto make_out =
      [&](Dtype odt,
          std::size_t elemSize) -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw)
      throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(odt, raw, N * elemSize));
  };

  // ---- Float32 ----
  if (isF32) {
    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float *>(outStore->data());
    const auto src = Xin.storage()->f32();
    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      dst[i] = std::fabs(src[lin]);
      if (!next_indexer(idx))
        break;
    }
    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Float64 ----
  if (isF64) {
    auto outStore = make_out(Dtype::Float64, sizeof(double));
    double *dst = reinterpret_cast<double *>(outStore->data());
    const auto src = Xin.storage()->f64();
    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      dst[i] = std::fabs(src[lin]);
      if (!next_indexer(idx))
        break;
    }
    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Int32 ----
  if (isI32) {
    auto outStore = make_out(Dtype::Int32, sizeof(std::int32_t));
    std::int32_t *dst = reinterpret_cast<std::int32_t *>(outStore->data());
    const auto src = Xin.storage()->i32();
    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const std::int32_t v = src[lin];
      // Guard INT32_MIN to avoid overflow on negation
      if (v == std::numeric_limits<std::int32_t>::min())
        throw std::runtime_error(fmt::format(
            "vkcnn: Abs \"{}\": abs(INT32_MIN) overflows.", node.name()));
      dst[i] = (v < 0) ? -v : v;
      if (!next_indexer(idx))
        break;
    }
    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Int64 ----
  if (isI64) {
    auto outStore = make_out(Dtype::Int64, sizeof(std::int64_t));
    std::int64_t *dst = reinterpret_cast<std::int64_t *>(outStore->data());
    const auto src = Xin.storage()->i64();
    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      const std::int64_t v = src[lin];
      if (v == std::numeric_limits<std::int64_t>::min())
        throw std::runtime_error(fmt::format(
            "vkcnn: Abs \"{}\": abs(INT64_MIN) overflows.", node.name()));
      dst[i] = (v < 0) ? -v : v;
      if (!next_indexer(idx))
        break;
    }
    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Sym ----
  {
    auto outStore = make_out(Dtype::Sym, sizeof(Sym));
    Sym *dst = reinterpret_cast<Sym *>(outStore->data());
    const auto src = Xin.storage()->sym();
    auto *sg = state.symGraph.get();

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lin = Xin.view().constIndexOf({idx.data(), idx.size()});
      dst[i] = sg->abs(src[lin]);
      if (!next_indexer(idx))
        break;
    }
    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }
}

} // namespace vkcnn::details
