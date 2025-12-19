#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> clip(
    ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() < 1 || inputs.size() > 3)
    throw std::runtime_error(fmt::format(
        "vkcnn: Clip \"{}\" expects 1..3 inputs (X, [min], [max]).", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Clip \"{}\" must have exactly 1 output.", nodeName));

  // ---- X ----
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Clip \"{}\": input X is required.", nodeName));
  const Tensor &X = *inputs[0];
  if (!X.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Clip \"{}\": only HostTensor is supported.", nodeName));
  const HostTensor &Xin = X.host();

  // Static shape + constant view (we rely on constIndexOf)
  if (!Xin.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Clip \"{}\": X must have constant shape.", nodeName));
  if (!Xin.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Clip \"{}\": X must have constant view.", nodeName));

  const TensorShape Xshape = Xin.shape();
  const auto dims = Xshape.toU64();
  std::size_t N = 1;
  for (auto d : dims)
    N *= static_cast<std::size_t>(d);

  const Dtype xdt = Xin.type();

  // ---- Helpers: read scalar host tensors ----
  auto is_scalar_len = [](const HostTensor &t) -> bool {
    if (!t.isConstant())
      return false;
    const auto d = t.shape().toU64();
    if (d.size() == 0)
      return true; // rank-0 scalar
    if (d.size() == 1 && d[0] == 1)
      return true; // 1-element vector
    return false;
  };

  auto read_scalar_index = [&](const HostTensor &t) -> std::size_t {
    // logical index -> storage index
    if (t.shape().rank() == 0) {
      // rank-0: constIndexOf ignores idx size (t.view() handles offset)
      return t.view().constIndexOf({});
    } else {
      std::uint64_t zero = 0;
      return t.view().constIndexOf({&zero, 1});
    }
  };

  // Float32/Float64 readers (strict dtype match)
  auto read_scalar_f32 = [&](const HostTensor &t) -> float {
    if (t.type() != Dtype::Float32)
      throw std::runtime_error("vkcnn: Clip: scalar must be Float32.");
    if (!is_scalar_len(t))
      throw std::runtime_error("vkcnn: Clip: scalar must be rank-0 or 1x1.");
    const auto idx = read_scalar_index(t);
    return t.storage()->f32()[idx];
  };
  auto read_scalar_f64 = [&](const HostTensor &t) -> double {
    if (t.type() != Dtype::Float64)
      throw std::runtime_error("vkcnn: Clip: scalar must be Float64.");
    if (!is_scalar_len(t))
      throw std::runtime_error("vkcnn: Clip: scalar must be rank-0 or 1x1.");
    const auto idx = read_scalar_index(t);
    return t.storage()->f64()[idx];
  };

  // Sym readers (allow Int64 promoted to Sym::Const)
  auto read_scalar_sym = [&](const HostTensor &t) -> Sym {
    if (!is_scalar_len(t))
      throw std::runtime_error("vkcnn: Clip: scalar must be rank-0 or 1x1.");
    const auto idx = read_scalar_index(t);
    if (t.type() == Dtype::Sym) {
      return t.storage()->sym()[idx];
    } else if (t.type() == Dtype::Int64) {
      return Sym::Const(t.storage()->i64()[idx]);
    } else {
      throw std::runtime_error(
          "vkcnn: Clip: for symbolic X, min/max must be SYM or INT64.");
    }
  };

  // ---- Optional min/max inputs ----
  const bool hasMin = (inputs.size() >= 2) && inputs[1].has_value();
  const bool hasMax = (inputs.size() >= 3) && inputs[2].has_value();

  // If present, they must be HostTensors
  const HostTensor *MinT = nullptr;
  const HostTensor *MaxT = nullptr;
  if (hasMin) {
    if (!inputs[1]->isHost())
      throw std::runtime_error("vkcnn: Clip: 'min' must be a HostTensor.");
    MinT = &inputs[1]->host();
  }
  if (hasMax) {
    if (!inputs[2]->isHost())
      throw std::runtime_error("vkcnn: Clip: 'max' must be a HostTensor.");
    MaxT = &inputs[2]->host();
  }

  // ---- Implement per dtype ----
  // We produce a new contiguous buffer with identity view+same shape.

  auto make_out =
      [&](Dtype dt,
          std::size_t elemSize) -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw)
      throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, N * elemSize));
  };

  // Helper to iterate source with const view
  auto next_indexer = [&](auto &idxVec) {
    // increments idxVec in row-major order given dims
    for (std::size_t ax = idxVec.size(); ax-- > 0;) {
      if (++idxVec[ax] < dims[ax])
        return true;
      idxVec[ax] = 0;
    }
    return false;
  };

  // FLOAT32
  if (xdt == Dtype::Float32) {
    float minv = -std::numeric_limits<float>::infinity();
    float maxv = +std::numeric_limits<float>::infinity();
    if (hasMin)
      minv = read_scalar_f32(*MinT);
    if (hasMax)
      maxv = read_scalar_f32(*MaxT);

    // Optional sanity: if both finite and min>max, it's ill-posed
    if (minv > maxv)
      throw std::runtime_error("vkcnn: Clip: min > max.");

    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float *>(outStore->data());

    if (Xin.isContiguous() && Xin.view().offset().isConstant() &&
        Xin.view().offset().constant() == 0) {
      const float *src = Xin.storage()->f32().data();
      for (std::size_t i = 0; i < N; ++i) {
        float v = src[i];
        if (hasMin && v < minv)
          v = minv;
        if (hasMax && v > maxv)
          v = maxv;
        dst[i] = v;
      }
    } else {
      const auto &view = Xin.view();
      const float *srcBase = Xin.storage()->f32().data();
      memory::vector<std::uint64_t> idx(dims.size(), 0);
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t lin = view.constIndexOf({idx.data(), idx.size()});
        float v = srcBase[lin];
        if (hasMin && v < minv)
          v = minv;
        if (hasMax && v > maxv)
          v = maxv;
        dst[i] = v;
        if (!next_indexer(idx))
          break;
      }
    }

    HostTensor out(Xshape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // FLOAT64
  if (xdt == Dtype::Float64) {
    double minv = -std::numeric_limits<double>::infinity();
    double maxv = +std::numeric_limits<double>::infinity();
    if (hasMin)
      minv = read_scalar_f64(*MinT);
    if (hasMax)
      maxv = read_scalar_f64(*MaxT);

    if (minv > maxv)
      throw std::runtime_error("vkcnn: Clip: min > max.");

    auto outStore = make_out(Dtype::Float64, sizeof(double));
    double *dst = reinterpret_cast<double *>(outStore->data());

    if (Xin.isContiguous() && Xin.view().offset().isConstant() &&
        Xin.view().offset().constant() == 0) {
      const double *src = Xin.storage()->f64().data();
      for (std::size_t i = 0; i < N; ++i) {
        double v = src[i];
        if (hasMin && v < minv)
          v = minv;
        if (hasMax && v > maxv)
          v = maxv;
        dst[i] = v;
      }
    } else {
      const auto &view = Xin.view();
      const double *srcBase = Xin.storage()->f64().data();
      memory::vector<std::uint64_t> idx(dims.size(), 0);
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t lin = view.constIndexOf({idx.data(), idx.size()});
        double v = srcBase[lin];
        if (hasMin && v < minv)
          v = minv;
        if (hasMax && v > maxv)
          v = maxv;
        dst[i] = v;
        if (!next_indexer(idx))
          break;
      }
    }

    HostTensor out(Xshape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // SYMBOLIC elements: elementwise y = min(max(x, min), max) using SymGraph
  if (xdt == Dtype::Sym) {
    Sym minS, maxS;
    const bool useMin = hasMin;
    const bool useMax = hasMax;
    if (useMin)
      minS = read_scalar_sym(*MinT);
    if (useMax)
      maxS = read_scalar_sym(*MaxT);

    auto outStore = make_out(Dtype::Sym, sizeof(Sym));
    Sym *dst = reinterpret_cast<Sym *>(outStore->data());

    auto g = state.symGraph;

    if (Xin.isContiguous() && Xin.view().offset().isConstant() &&
        Xin.view().offset().constant() == 0) {
      const auto src = Xin.storage()->sym();
      for (std::size_t i = 0; i < N; ++i) {
        Sym v = src[i];
        if (useMin)
          v = g->max(v, minS);
        if (useMax)
          v = g->min(v, maxS);
        dst[i] = v;
      }
    } else {
      const auto &view = Xin.view();
      const auto src = Xin.storage()->sym();
      memory::vector<std::uint64_t> idx(dims.size(), 0);
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t lin = view.constIndexOf({idx.data(), idx.size()});
        Sym v = src[lin];
        if (useMin)
          v = g->max(v, minS);
        if (useMax)
          v = g->min(v, maxS);
        dst[i] = v;
        if (!next_indexer(idx))
          break;
      }
    }

    HostTensor out(Xshape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }

  // If you want, you can add int dtypes here similarly.
  throw std::runtime_error(
      fmt::format("vkcnn: Clip \"{}\": unsupported dtype {} for host Clip.",
                  nodeName, xdt.to_string()));
}

} // namespace denox::onnx::details::ops
