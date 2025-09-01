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
import_op_Mul(ImportState &state, std::span<const std::optional<Tensor>> inputs,
              std::size_t outputCount,
              const std::unordered_map<std::string, Attribute> & /*attributes*/,
              [[maybe_unused]] opset_version /*version*/,
              const onnx::NodeProto &node) {

  // Arity
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Mul \"{}\" expects 2 inputs.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Mul \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &aT = *inputs[0];
  const Tensor &bT = *inputs[1];

  // Runtime tensors not supported
  if (aT.isDevice() || bT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Mul \"{}\": runtime tensors not supported.", node.name()));

  const HostTensor &a0 = aT.host();
  const HostTensor &b0 = bT.host();

  // Must be static (we rely on constIndexOf)
  if (!a0.isConstant() || !b0.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Mul \"{}\": dynamic host tensors unsupported.", node.name()));

  const Dtype adt = a0.type();
  const Dtype bdt = b0.type();

  auto is_float32_or_64 = [](Dtype dt) {
    return dt == Dtype::Float32 || dt == Dtype::Float64;
  };
  auto is_signed_int = [](Dtype dt) {
    switch (dt) {
    case Dtype::Int8:
    case Dtype::Int16:
    case Dtype::Int32:
    case Dtype::Int64:
      return true;
    default:
      return false;
    }
  };
  auto is_unsigned_int = [](Dtype dt) {
    switch (dt) {
    case Dtype::Uint8:
    case Dtype::Uint16:
    case Dtype::Uint32:
    case Dtype::Uint64:
      return true;
    default:
      return false;
    }
  };
  auto is_integer = [&](Dtype dt) {
    return is_signed_int(dt) || is_unsigned_int(dt);
  };

  // Disallow strings/bools outright (and we also do not handle Float16 here)
  if (adt == Dtype::String || bdt == Dtype::String || adt == Dtype::Bool ||
      bdt == Dtype::Bool || adt == Dtype::Float16 || bdt == Dtype::Float16)
    throw std::runtime_error(fmt::format(
        "vkcnn: Mul \"{}\": unsupported dtype (string/bool/float16).",
        node.name()));

  // Make base-contiguous
  HostTensor A = a0.contiguous();
  HostTensor B = b0.contiguous();

  // Broadcast shape & views
  TensorShape outShape = TensorShape::broadcast(A.shape(), B.shape());
  const auto outDims = outShape.toU64();
  const size_t outRank = outDims.size();

  auto make_axes = [&](size_t rIn) {
    std::vector<int64_t> m(rIn);
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
        "vkcnn: Mul \"{}\": non-constant broadcast view.", node.name()));

  // N-D iteration
  std::vector<std::uint64_t> idx(outRank, 0);
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

  // Load helpers
  auto load_int64 = [&](const HostTensor &H,
                        const TensorViewDesc &V) -> int64_t {
    const std::size_t k = V.constIndexOf({idx.data(), idx.size()});
    switch (H.type()) {
    case Dtype::Int8:
      return static_cast<int64_t>(
          static_cast<const int8_t *>(H.storage()->data())[k]);
    case Dtype::Int16:
      return static_cast<int64_t>(
          static_cast<const int16_t *>(H.storage()->data())[k]);
    case Dtype::Int32:
      return static_cast<int64_t>(
          static_cast<const int32_t *>(H.storage()->data())[k]);
    case Dtype::Int64:
      return static_cast<int64_t>(
          static_cast<const int64_t *>(H.storage()->data())[k]);
    case Dtype::Uint8:
      return static_cast<int64_t>(
          static_cast<const uint8_t *>(H.storage()->data())[k]);
    case Dtype::Uint16:
      return static_cast<int64_t>(
          static_cast<const uint16_t *>(H.storage()->data())[k]);
    case Dtype::Uint32:
      return static_cast<int64_t>(
          static_cast<const uint32_t *>(H.storage()->data())[k]);
    case Dtype::Uint64:
      return static_cast<int64_t>(
          static_cast<const uint64_t *>(H.storage()->data())[k]);
    default:
      throw std::logic_error("load_int64: non-integer dtype");
    }
  };
  auto load_uint64 = [&](const HostTensor &H,
                         const TensorViewDesc &V) -> uint64_t {
    const std::size_t k = V.constIndexOf({idx.data(), idx.size()});
    switch (H.type()) {
    case Dtype::Int8:
      return static_cast<uint64_t>(
          static_cast<const int8_t *>(H.storage()->data())[k]);
    case Dtype::Int16:
      return static_cast<uint64_t>(
          static_cast<const int16_t *>(H.storage()->data())[k]);
    case Dtype::Int32:
      return static_cast<uint64_t>(
          static_cast<const int32_t *>(H.storage()->data())[k]);
    case Dtype::Int64:
      return static_cast<uint64_t>(
          static_cast<const int64_t *>(H.storage()->data())[k]);
    case Dtype::Uint8:
      return static_cast<uint64_t>(
          static_cast<const uint8_t *>(H.storage()->data())[k]);
    case Dtype::Uint16:
      return static_cast<uint64_t>(
          static_cast<const uint16_t *>(H.storage()->data())[k]);
    case Dtype::Uint32:
      return static_cast<uint64_t>(
          static_cast<const uint32_t *>(H.storage()->data())[k]);
    case Dtype::Uint64:
      return static_cast<uint64_t>(
          static_cast<const uint64_t *>(H.storage()->data())[k]);
    default:
      throw std::logic_error("load_uint64: non-integer dtype");
    }
  };
  auto load_double = [&](const HostTensor &H,
                         const TensorViewDesc &V) -> double {
    const std::size_t k = V.constIndexOf({idx.data(), idx.size()});
    switch (H.type()) {
    case Dtype::Float32:
      return static_cast<const float *>(H.storage()->data())[k];
    case Dtype::Float64:
      return static_cast<const double *>(H.storage()->data())[k];
    case Dtype::Int8:
      return static_cast<const int8_t *>(H.storage()->data())[k];
    case Dtype::Int16:
      return static_cast<const int16_t *>(H.storage()->data())[k];
    case Dtype::Int32:
      return static_cast<const int32_t *>(H.storage()->data())[k];
    case Dtype::Int64:
      return static_cast<const int64_t *>(H.storage()->data())[k];
    case Dtype::Uint8:
      return static_cast<const uint8_t *>(H.storage()->data())[k];
    case Dtype::Uint16:
      return static_cast<const uint16_t *>(H.storage()->data())[k];
    case Dtype::Uint32:
      return static_cast<const uint32_t *>(H.storage()->data())[k];
    case Dtype::Uint64:
      return static_cast<const uint64_t *>(H.storage()->data())[k];
    default:
      throw std::logic_error("load_double: unsupported dtype");
    }
  };

  // -------- Symbolic path (only with integers) --------
  if (adt == Dtype::Sym || bdt == Dtype::Sym) {
    if (is_float32_or_64(adt) || is_float32_or_64(bdt))
      throw std::runtime_error(
          fmt::format("vkcnn: Mul \"{}\": symbolic with floats not supported.",
                      node.name()));

    auto load_sym = [&](const HostTensor &H, const TensorViewDesc &V) -> Sym {
      const std::size_t k = V.constIndexOf({idx.data(), idx.size()});
      switch (H.type()) {
      case Dtype::Sym:
        return static_cast<const Sym *>(H.storage()->data())[k];
      case Dtype::Int8:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const int8_t *>(H.storage()->data())[k]));
      case Dtype::Int16:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const int16_t *>(H.storage()->data())[k]));
      case Dtype::Int32:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const int32_t *>(H.storage()->data())[k]));
      case Dtype::Int64:
        return Sym::Const(static_cast<const int64_t *>(H.storage()->data())[k]);
      case Dtype::Uint8:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const uint8_t *>(H.storage()->data())[k]));
      case Dtype::Uint16:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const uint16_t *>(H.storage()->data())[k]));
      case Dtype::Uint32:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const uint32_t *>(H.storage()->data())[k]));
      case Dtype::Uint64:
        return Sym::Const(static_cast<int64_t>(
            static_cast<const uint64_t *>(H.storage()->data())[k]));
      default:
        throw std::logic_error("load_sym: invalid dtype");
      }
    };

    std::size_t outCount = 1;
    for (auto d : outDims)
      outCount *= (size_t)d;
    void *rawOut = std::malloc(outCount * sizeof(Sym));
    if (!rawOut)
      throw std::bad_alloc();
    Sym *po = static_cast<Sym *>(rawOut);

    while (true) {
      const Sym xs = load_sym(A, viewA);
      const Sym ys = load_sym(B, viewB);
      Sym r = state.symGraph->mul(xs, ys); // integer symbolic multiply
      *po++ = r;
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Sym, rawOut, outCount * sizeof(Sym)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- Float path (any float present → float) --------
  if (is_float32_or_64(adt) || is_float32_or_64(bdt)) {
    const Dtype rdt = (adt == Dtype::Float64 || bdt == Dtype::Float64)
                          ? Dtype::Float64
                          : Dtype::Float32;
    const size_t elem = dtype_size(rdt);
    std::size_t outCount = 1;
    for (auto d : outDims)
      outCount *= (size_t)d;
    void *rawOut = std::malloc(outCount * elem);
    if (!rawOut)
      throw std::bad_alloc();

    if (rdt == Dtype::Float64) {
      double *po = static_cast<double *>(rawOut);
      while (true) {
        const double x = load_double(A, viewA);
        const double y = load_double(B, viewB);
        *po++ = (x * y);
        if (!inc())
          break;
      }
    } else {
      float *po = static_cast<float *>(rawOut);
      while (true) {
        const double x = load_double(A, viewA);
        const double y = load_double(B, viewB);
        *po++ = static_cast<float>(x * y);
        if (!inc())
          break;
      }
    }

    auto outStore = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(rdt, rawOut, outCount * elem));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }

  // -------- Integer path (mixed widths/signedness allowed) --------
  if (!(is_integer(adt) && is_integer(bdt)))
    throw std::runtime_error(fmt::format(
        "vkcnn: Mul \"{}\": unsupported dtype combination.", node.name()));

  const bool anySigned = is_signed_int(adt) || is_signed_int(bdt);
  std::size_t outCount = 1;
  for (auto d : outDims)
    outCount *= (size_t)d;

  if (anySigned) {
    void *rawOut = std::malloc(outCount * sizeof(int64_t));
    if (!rawOut)
      throw std::bad_alloc();
    auto *po = static_cast<int64_t *>(rawOut);

    while (true) {
      const int64_t x = load_int64(A, viewA);
      const int64_t y = load_int64(B, viewB);
      *po++ = (x * y); // overflow ignored per your policy
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
      const uint64_t x = load_uint64(A, viewA);
      const uint64_t y = load_uint64(B, viewB);
      *po++ = (x * y); // wrap-around allowed
      if (!inc())
        break;
    }

    auto outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Uint64, rawOut, outCount * sizeof(uint64_t)));
    return {Tensor::Host(HostTensor(outShape, std::move(outStore)))};
  }
}

} // namespace vkcnn::details
