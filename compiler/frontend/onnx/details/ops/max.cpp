#include "frontend/onnx/details/ops/ops.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
max(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // ---- arity ----
  if (inputs.empty())
    throw std::runtime_error(
        fmt::format("vkcnn: Max \"{}\" expects at least 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Max \"{}\" must have exactly 1 output.", nodeName));

  // ---- gather & validate inputs (host-only) ----
  memory::vector<const HostTensor *> Xs;
  Xs.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value())
      throw std::runtime_error(
          fmt::format("vkcnn: Max \"{}\": input {} is missing.", nodeName, i));
    const Tensor &t = *inputs[i];
    if (!t.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Max \"{}\": only HostTensor is supported (input {}).",
          nodeName, i));
    const HostTensor &h = t.host();
    if (!h.isConstant())
      throw std::runtime_error(
          fmt::format("vkcnn: Max \"{}\": input {} must have constant shape.",
                      nodeName, i));
    if (!h.view().isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: Max \"{}\": input {} must have constant view.", nodeName, i));
    Xs.push_back(&h);
  }

  // ---- dtype checks ----
  const Dtype dt0 = Xs[0]->type();
  auto all_same_dtype =
      std::all_of(Xs.begin(), Xs.end(),
                  [&](const HostTensor *h) { return h->type() == dt0; });
  if (!all_same_dtype) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Max \"{}\": all inputs must have the same dtype.", nodeName));
  }

  const bool isF32 = (dt0 == Dtype::Float32);
  const bool isF64 = (dt0 == Dtype::Float64);
  const bool isSym = (dt0 == Dtype::Sym);

  if (!isF32 && !isF64 && !isSym) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Max \"{}\": unsupported dtype {} (only Float32, Float64, Sym).",
        nodeName, dt0.to_string()));
  }

  // ---- compute broadcasted output shape ----
  TensorShape outShape = Xs[0]->shape();
  for (std::size_t i = 1; i < Xs.size(); ++i) {
    outShape = TensorShape::broadcast(outShape, Xs[i]->shape());
  }
  const auto outDims =
      outShape.toU64(); // outShape must be constant due to inputs constant
  std::size_t N = 1;
  for (auto d : outDims)
    N *= static_cast<std::size_t>(d);

  // ---- build broadcasted views for each input ----
  // Axes map: align trailing dims (NumPy). For input rank r_in, out rank r_out:
  // axesMap[i] = r_out - r_in + i, for i in [0..r_in-1]
  struct BView {
    TensorViewDesc view;
  };
  memory::vector<BView> views;
  views.reserve(Xs.size());

  for (const HostTensor *h : Xs) {
    const TensorShape inShape = h->shape();
    const std::size_t rin = inShape.rank();
    const std::size_t rout = outShape.rank();

    memory::vector<int64_t> axesMap(rin);
    for (std::size_t i = 0; i < rin; ++i) {
      axesMap[i] = static_cast<int64_t>(rout - rin + i);
    }
    TensorViewDesc bview =
        h->view().broadcastInDim(inShape.dims(), outShape.dims(), axesMap);
    views.push_back(BView{std::move(bview)});
  }

  // ---- allocate output storage ----
  auto make_out =
      [&](Dtype dt,
          std::size_t elemSize) -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw)
      throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, N * elemSize));
  };

  // ---- index iteration helper ----
  auto next_indexer = [&](auto &idxVec) {
    for (std::size_t ax = idxVec.size(); ax-- > 0;) {
      if (++idxVec[ax] < outDims[ax])
        return true;
      idxVec[ax] = 0;
    }
    return false;
  };

  // ---- Float32 ----
  if (isF32) {
    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float *>(outStore->data());

    // Pre-take base storages
    memory::vector<const float *> bases;
    bases.reserve(Xs.size());
    for (auto *h : Xs)
      bases.push_back(h->storage()->f32().data());

    memory::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      // compute max across inputs at idx
      float mv;
      {
        const std::size_t lin0 =
            views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link =
            views[k].view.constIndexOf({idx.data(), idx.size()});
        float v = bases[k][link];
        if (v > mv)
          mv = v;
      }
      dst[i] = mv;
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

    memory::vector<const double *> bases;
    bases.reserve(Xs.size());
    for (auto *h : Xs)
      bases.push_back(h->storage()->f64().data());

    memory::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      double mv;
      {
        const std::size_t lin0 =
            views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link =
            views[k].view.constIndexOf({idx.data(), idx.size()});
        double v = bases[k][link];
        if (v > mv)
          mv = v;
      }
      dst[i] = mv;
      if (!next_indexer(idx))
        break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // ---- Sym (symbolic element type) ----
  if (isSym) {
    auto outStore = make_out(Dtype::Sym, sizeof(Sym));
    Sym *dst = reinterpret_cast<Sym *>(outStore->data());

    memory::vector<const Sym *> bases;
    bases.reserve(Xs.size());
    for (auto *h : Xs)
      bases.push_back(h->storage()->sym().data());

    auto sg = state.symGraph;

    memory::vector<std::uint64_t> idx(outDims.size(), 0);
    for (std::size_t i = 0; i < N; ++i) {
      Sym mv;
      {
        const std::size_t lin0 =
            views[0].view.constIndexOf({idx.data(), idx.size()});
        mv = bases[0][lin0];
      }
      for (std::size_t k = 1; k < Xs.size(); ++k) {
        const std::size_t link =
            views[k].view.constIndexOf({idx.data(), idx.size()});
        Sym v = bases[k][link];
        mv = sg->max(mv, v);
      }
      dst[i] = mv;
      if (!next_indexer(idx))
        break;
    }

    HostTensor out(outShape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // unreachable for now
  throw std::runtime_error(
      "vkcnn: Max: internal error: dtype dispatch failed.");
}

} // namespace denox::onnx::details::ops
