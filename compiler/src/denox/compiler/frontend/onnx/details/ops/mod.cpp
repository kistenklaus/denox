#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <cmath>
#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
mod(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // ---- arity ----
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Mod \"{}\" expects 2 inputs.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Mod \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &aT = *inputs[0];
  const Tensor &bT = *inputs[1];

  // Host-only for now
  if (aT.isDevice() || bT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Mod \"{}\": runtime tensors not supported.", nodeName));

  const HostTensor &a0 = aT.host();
  const HostTensor &b0 = bT.host();

  // We rely on constIndexOf during iteration
  if (!a0.isConstant() || !b0.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Mod \"{}\": dynamic host tensors unsupported.", nodeName));

  const Dtype adt = a0.type();
  const Dtype bdt = b0.type();

  // quick dtype flags
  const bool aIsFloat = adt.isFloat();
  const bool bIsFloat = bdt.isFloat();
  const bool aIsInt = adt.isInteger();
  const bool bIsInt = bdt.isInteger();
  const bool aIsSym = (adt == Dtype::Sym);
  const bool bIsSym = (bdt == Dtype::Sym);

  // Make base-contiguous for clean reads
  HostTensor A = a0.contiguous();
  HostTensor B = b0.contiguous();

  // Broadcast and views
  TensorShape outShape = TensorShape::broadcast(A.shape(), B.shape());
  const auto outDims = outShape.toU64();
  const size_t outRank = outDims.size();

  auto make_axes = [&](size_t rIn) {
    memory::vector<int64_t> m(rIn);
    const int64_t shift = static_cast<int64_t>(outRank - rIn);
    for (size_t i = 0; i < rIn; ++i)
      m[i] = shift + static_cast<int64_t>(i);
    return m;
  };
  const auto axesA = make_axes(A.rank());
  const auto axesB = make_axes(B.rank());

  TensorViewDesc viewA =
      A.view().broadcastInDim(A.shape().dims(), outShape.dims(), axesA);
  TensorViewDesc viewB =
      B.view().broadcastInDim(B.shape().dims(), outShape.dims(), axesB);

  if (!viewA.isConstant() || !viewB.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Mod \"{}\": non-constant broadcast view.", nodeName));

  HostTensor A_broadcasted = A.withView(outShape, viewA);
  HostTensor B_broadcasted = B.withView(outShape, viewB);

  // N-D index iteration
  memory::vector<std::uint64_t> idx(outRank, 0);
  auto inc = [&]() -> bool {
    if (idx.empty())
      return false;
    size_t ax = outRank;
    while (ax > 0) {
      --ax;
      if (++idx[ax] < outDims[ax])
        return true;
      idx[ax] = 0;
    }
    return false;
  };

  // total elements
  std::size_t outCount = 1;
  for (auto d : outDims)
    outCount *= static_cast<std::size_t>(d);

  // -------- FLOATS (use fmod or euclidean remainder) --------
  if (aIsFloat || bIsFloat) {
    if (aIsSym || bIsSym)
      throw std::runtime_error(fmt::format(
          "vkcnn: Mod \"{}\": symbolic with floats not supported.", nodeName));

    // ONNX attr fmod (int). If present ==0 → Euclidean; else → fmod.
    bool useFmod = true; // default
    if (auto it = attributes.find("fmod"); it != attributes.end()) {
      if (!it->second.isInt())
        throw std::runtime_error(fmt::format(
            "vkcnn: Mod \"{}\": attribute 'fmod' must be int.", nodeName));
      useFmod = (it->second.i() != 0);
    }

    const Dtype rdt = (adt == Dtype::Float64 || bdt == Dtype::Float64)
                          ? Dtype::Float64
                          : Dtype::Float32;
    const std::size_t elem = rdt.size();

    void *rawOut = std::malloc(outCount * elem);
    if (!rawOut)
      throw std::bad_alloc();

    if (rdt == Dtype::Float64) {
      double *po = static_cast<double *>(rawOut);
      while (true) {
        const double x = A_broadcasted.loadDouble(idx);
        const double y = B_broadcasted.loadDouble(idx);
        if (y == 0.0)
          throw std::runtime_error(fmt::format(
              "vkcnn: Mod \"{}\": division by zero (float).", nodeName));
        const double r =
            useFmod ? std::fmod(x, y) : (x - y * std::floor(x / y));
        *po++ = r;
        if (!inc())
          break;
      }
    } else {
      float *po = static_cast<float *>(rawOut);
      while (true) {
        const double x = A_broadcasted.loadDouble(idx);
        const double y = B_broadcasted.loadDouble(idx);
        if (y == 0.0)
          throw std::runtime_error(fmt::format(
              "vkcnn: Mod \"{}\": division by zero (float).", nodeName));
        const double r =
            useFmod ? std::fmod(x, y) : (x - y * std::floor(x / y));
        *po++ = static_cast<float>(r);
        if (!inc())
          break;
      }
    }

    auto outStore = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(rdt, rawOut, outCount * elem));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- SYMBOLIC (no floats allowed) --------
  if (aIsSym || bIsSym) {
    // Only Sym with (Sym or Int). No floats already ensured.
    void *rawOut = std::malloc(outCount * sizeof(Sym));
    if (!rawOut)
      throw std::bad_alloc();
    auto *po = static_cast<Sym *>(rawOut);

    while (true) {
      const Sym xs = A_broadcasted.loadSym(idx);
      const Sym ys = B_broadcasted.loadSym(idx);

      if (ys.isConstant() && ys.constant() == 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Mod \"{}\": division by zero (sym).", nodeName));

      // If exactly one side is constant and negative → forbid (your earlier
      // rule)
      const bool xC = xs.isConstant();
      const bool yC = ys.isConstant();
      if (xC != yC) {
        const auto neg = (xC ? xs.constant() : ys.constant()) < 0;
        if (neg) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Mod \"{}\": negative constant with symbolic counterpart "
              "is not supported.",
              nodeName));
        }
      }

      *po++ = state.symGraph->mod(xs, ys); // Euclidean mod in your engine
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Sym, rawOut, outCount * sizeof(Sym)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- INTEGERS --------
  if (!(aIsInt && bIsInt))
    throw std::runtime_error(fmt::format(
        "vkcnn: Mod \"{}\": unsupported dtype combination.", nodeName));

  const bool anySigned = adt.isSignedInt() || bdt.isSignedInt();

  if (anySigned) {
    void *rawOut = std::malloc(outCount * sizeof(int64_t));
    if (!rawOut)
      throw std::bad_alloc();
    auto *po = static_cast<int64_t *>(rawOut);

    while (true) {
      const int64_t x = A_broadcasted.loadI64(idx);
      const int64_t y = B_broadcasted.loadI64(idx);
      if (y == 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Mod \"{}\": division by zero (int).", nodeName));
      *po++ = (x % y); // C++ % for signed (you earlier said assume positives)
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Int64, rawOut, outCount * sizeof(int64_t)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  } else {
    void *rawOut = std::malloc(outCount * sizeof(uint64_t));
    if (!rawOut)
      throw std::bad_alloc();
    auto *po = static_cast<uint64_t *>(rawOut);

    while (true) {
      const uint64_t x = A_broadcasted.loadU64(idx);
      const uint64_t y = B_broadcasted.loadU64(idx);
      if (y == 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Mod \"{}\": division by zero (int).", nodeName));
      *po++ = (x % y); // wrap-around semantics ok for unsigned
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Uint64, rawOut, outCount * sizeof(uint64_t)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }
}

} // namespace denox::onnx::details::ops
