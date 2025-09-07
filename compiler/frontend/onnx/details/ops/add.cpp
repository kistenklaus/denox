#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <span>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
add(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // Arity
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Add \"{}\" expects 2 inputs.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Add \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &aT = *inputs[0];
  const Tensor &bT = *inputs[1];

  // Runtime tensors not supported here
  if (aT.isDevice() || bT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Add \"{}\": runtime tensors not supported.", nodeName));

  const HostTensor &a0 = aT.host();
  const HostTensor &b0 = bT.host();

  // Must be static for host compute (we rely on constIndexOf)
  if (!a0.isConstant() || !b0.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Add \"{}\": dynamic host tensors unsupported.", nodeName));

  const Dtype adt = a0.type();
  const Dtype bdt = b0.type();

  // Disallow strings/bools outright
  if (adt == Dtype::String || bdt == Dtype::String || adt == Dtype::Bool ||
      bdt == Dtype::Bool)
    throw std::runtime_error(fmt::format(
        "vkcnn: Add \"{}\": unsupported dtype (string/bool).", nodeName));

  // Make base-contiguous for clean reads
  HostTensor A = a0.contiguous();
  HostTensor B = b0.contiguous();

  // Broadcast shape & build broadcast views
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
        "vkcnn: Add \"{}\": non-constant broadcast view.", nodeName));

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

  // -------- Symbolic path (only with integers) --------
  if (adt == Dtype::Sym || bdt == Dtype::Sym) {
    if (adt.isFloat() || bdt.isFloat())
      throw std::runtime_error(fmt::format(
          "vkcnn: Add \"{}\": symbolic with floats not supported.", nodeName));

    std::size_t outCount = 1;
    for (auto d : outDims)
      outCount *= static_cast<std::size_t>(d);
    void *rawOut = std::malloc(outCount * sizeof(compiler::Sym));
    if (!rawOut)
      throw std::bad_alloc();
    compiler::Sym *po = static_cast<compiler::Sym *>(rawOut);

    while (true) {
      const compiler::Sym xs = A_broadcasted.loadSym(idx);
      const compiler::Sym ys = B_broadcasted.loadSym(idx);
      compiler::Sym r =
          state.symGraph->add(xs, ys); // no positivity assumptions
      *po++ = r;
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Sym, rawOut, outCount * sizeof(compiler::Sym)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- Float path (any float present → float; ints are promoted) --------
  if (adt.isFloat() || bdt.isFloat()) {
    const Dtype rdt = (adt == Dtype::Float64 || bdt == Dtype::Float64)
                          ? Dtype::Float64
                          : Dtype::Float32;
    const size_t elem = rdt.size();
    std::size_t outCount = 1;
    for (auto d : outDims)
      outCount *= static_cast<std::size_t>(d);
    void *rawOut = std::malloc(outCount * elem);
    if (!rawOut)
      throw std::bad_alloc();

    if (rdt == Dtype::Float64) {
      double *po = static_cast<double *>(rawOut);
      while (true) {
        const double x = A_broadcasted.loadDouble(idx);
        const double y = B_broadcasted.loadDouble(idx);
        *po++ = (x + y);
        if (!inc())
          break;
      }
    } else {
      float *po = static_cast<float *>(rawOut);
      while (true) {
        const double x = A_broadcasted.loadDouble(idx);
        const double y = B_broadcasted.loadDouble(idx);
        *po++ = static_cast<float>(x + y);
        if (!inc())
          break;
      }
    }

    auto outStore = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(rdt, rawOut, outCount * elem));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- Integer path (mixed widths/signedness allowed) --------
  if (!(adt.isInteger() && bdt.isInteger()))
    throw std::runtime_error(fmt::format(
        "vkcnn: Add \"{}\": unsupported dtype combination.", nodeName));

  const bool anySigned = adt.isSignedInt() || bdt.isSignedInt();
  std::size_t outCount = 1;
  for (auto d : outDims)
    outCount *= static_cast<std::size_t>(d);

  if (anySigned) {
    void *rawOut = std::malloc(outCount * sizeof(int64_t));
    if (!rawOut)
      throw std::bad_alloc();
    auto *po = static_cast<int64_t *>(rawOut);

    while (true) {
      const int64_t x = A_broadcasted.loadI64(idx);
      const int64_t y = B_broadcasted.loadI64(idx);
      *po++ = (x + y); 
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
      *po++ = (x + y); // wrap-around allowed
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
