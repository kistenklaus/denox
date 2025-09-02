#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Min(
    ImportState &state,
    std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> & /*attributes*/,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // ---- arity ----
  if (inputs.empty())
    throw std::runtime_error(
        fmt::format("vkcnn: Min \"{}\" expects at least 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Min \"{}\" must have exactly 1 output.", node.name()));

  // ---- gather & validate inputs (host-only) ----
  std::vector<const HostTensor*> Xs;
  Xs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value())
      throw std::runtime_error(fmt::format(
          "vkcnn: Min \"{}\": input {} is missing.", node.name(), i));
    const Tensor &t = *inputs[i];
    if (!t.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Min \"{}\": only HostTensor is supported (input {}).",
          node.name(), i));
    const HostTensor &h = t.host();
    if (!h.isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: Min \"{}\": input {} must have constant shape.", node.name(), i));
    if (!h.view().isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: Min \"{}\": input {} must have constant view.", node.name(), i));
    Xs.push_back(&h);
  }

  // ---- dtype checks ----
  const Dtype dt0 = Xs[0]->type();
  auto all_same_dtype = std::all_of(Xs.begin(), Xs.end(),
                                    [&](const HostTensor* h){ return h->type() == dt0; });
  if (!all_same_dtype)
    throw std::runtime_error(fmt::format(
        "vkcnn: Min \"{}\": all inputs must have the same dtype.", node.name()));

  const bool isF32 = (dt0 == Dtype::Float32);
  const bool isF64 = (dt0 == Dtype::Float64);
  const bool isSym = (dt0 == Dtype::Sym);
  if (!isF32 && !isF64 && !isSym)
    throw std::runtime_error(fmt::format(
        "vkcnn: Min \"{}\": unsupported dtype {} (only Float32, Float64, Sym).",
        node.name(), dtype_to_string(dt0)));

  // ---- broadcast output shape ----
  TensorShape outShape = Xs[0]->shape();
  for (size_t i = 1; i < Xs.size(); ++i)
    outShape = TensorShape::broadcast(outShape, Xs[i]->shape());
  const auto outDims = outShape.toU64();
  std::size_t N = 1; for (auto d : outDims) N *= (std::size_t)d;

  // ---- build broadcasted views ----
  struct BView { TensorViewDesc view; };
  std::vector<BView> views; views.reserve(Xs.size());
  for (const HostTensor* h : Xs) {
    const TensorShape inShape = h->shape();
    const size_t rin = inShape.rank();
    const size_t rout = outShape.rank();
    std::vector<int64_t> axesMap(rin);
    for (size_t i = 0; i < rin; ++i)
      axesMap[i] = (int64_t)(rout - rin + i);
    views.push_back(BView{
      h->view().broadcastInDim(inShape.dims(), outShape.dims(), axesMap)
    });
  }

  // ---- alloc helper ----
  auto make_out = [&](Dtype dt, std::size_t elemSize)
      -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw) throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, N * elemSize));
  };

  // ---- index iterator ----
  auto next_indexer = [&](auto &idxVec) {
    for (std::size_t ax = idxVec.size(); ax-- > 0;) {
      if (++idxVec[ax] < outDims[ax]) return true;
      idxVec[ax] = 0;
    }
    return false;
  };

  // ---- Float32 ----
  if (isF32) {
    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float*>(outStore->data());

    std::vector<const float*> bases;
    bases.reserve(Xs.size());
    for (auto* h : Xs) bases.push_back(h->storage()->f32().data());

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      float mv;
      {
        const std::size_t lin0 = views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link = views[k].view.constIndexOf({idx.data(), idx.size()});
        float v = bases[k][link];
        if (v < mv) mv = v;
      }
      dst[i] = mv;
      if (!next_indexer(idx)) break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Float64 ----
  if (isF64) {
    auto outStore = make_out(Dtype::Float64, sizeof(double));
    double *dst = reinterpret_cast<double*>(outStore->data());

    std::vector<const double*> bases;
    bases.reserve(Xs.size());
    for (auto* h : Xs) bases.push_back(h->storage()->f64().data());

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      double mv;
      {
        const std::size_t lin0 = views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link = views[k].view.constIndexOf({idx.data(), idx.size()});
        double v = bases[k][link];
        if (v < mv) mv = v;
      }
      dst[i] = mv;
      if (!next_indexer(idx)) break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Sym ----
  {
    auto outStore = make_out(Dtype::Sym, sizeof(Sym));
    Sym *dst = reinterpret_cast<Sym*>(outStore->data());

    std::vector<const Sym*> bases;
    bases.reserve(Xs.size());
    for (auto* h : Xs) bases.push_back(h->storage()->sym().data());

    auto *sg = state.symGraph.get();

    std::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      Sym mv;
      {
        const std::size_t lin0 = views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link = views[k].view.constIndexOf({idx.data(), idx.size()});
        Sym v = bases[k][link];
        mv = sg->min(mv, v);
      }
      dst[i] = mv;
      if (!next_indexer(idx)) break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }
}

} // namespace vkcnn::details
