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
import_op_Sub(ImportState &state, std::span<const std::optional<Tensor>> inputs,
              std::size_t outputCount,
              const std::unordered_map<std::string, Tensor> &attributes,
              opset_version /*version*/, const onnx::NodeProto &node) {

  // --- contract
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Sub requires exactly two inputs. Got {} (node='{}')",
        inputs.size(), node.op_type()));
  }
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Sub must produce exactly one output");
  }
  const Tensor &A = *inputs[0];
  const Tensor &B = *inputs[1];

  // --- early rejects
  if (A.isRuntimeTensor() || B.isRuntimeTensor()) {
    throw std::runtime_error("vkcnn: Sub on runtime tensors is not supported");
  }
  if (A.isString() || B.isString()) {
    throw std::runtime_error("vkcnn: Sub on string tensors is not supported");
  }

  // --- helpers: scalar lifts (from ScalarTensor or RawTensor(0D)) ----------
  auto scalar_int64_from_scalar =
      [](const Tensor &t) -> std::optional<int64_t> {
    if (!t.isScalar())
      return std::nullopt;
    const auto &s = t.scalar();
    switch (s.dtype) {
    case Dtype::Int64:
      return s.v.i;
    case Dtype::Int32:
    case Dtype::Int16:
    case Dtype::Int8:
      return static_cast<int64_t>(s.v.i);
    case Dtype::Uint64:
      if (s.v.u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error(
            "vkcnn: Sub uint64 scalar too large for int64");
      return static_cast<int64_t>(s.v.u);
    case Dtype::Uint32:
    case Dtype::Uint16:
    case Dtype::Uint8:
      return static_cast<int64_t>(s.v.u);
    default:
      return std::nullopt;
    }
  };
  auto scalar_int64_from_raw0 = [](const Tensor &t) -> std::optional<int64_t> {
    if (!t.isRaw())
      return std::nullopt;
    const auto &rt = t.raw();
    if (!rt.shape.isScalar())
      return std::nullopt;
    auto need = [&](size_t n) {
      if (rt.raw.size() != n)
        throw std::runtime_error("vkcnn: Sub raw scalar payload size mismatch");
    };
    switch (rt.type) {
    case Dtype::Int64: {
      need(sizeof(int64_t));
      int64_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return v;
    }
    case Dtype::Int32: {
      need(sizeof(int32_t));
      int32_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    case Dtype::Int16: {
      need(sizeof(int16_t));
      int16_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    case Dtype::Int8: {
      need(sizeof(int8_t));
      int8_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    case Dtype::Uint64: {
      need(sizeof(uint64_t));
      uint64_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      if (v > (uint64_t)std::numeric_limits<int64_t>::max())
        throw std::runtime_error(
            "vkcnn: Sub raw uint64 scalar too large for int64");
      return (int64_t)v;
    }
    case Dtype::Uint32: {
      need(sizeof(uint32_t));
      uint32_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    case Dtype::Uint16: {
      need(sizeof(uint16_t));
      uint16_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    case Dtype::Uint8: {
      need(sizeof(uint8_t));
      uint8_t v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return (int64_t)v;
    }
    default:
      return std::nullopt;
    }
  };
  struct FScalar {
    bool is64;
    double v;
  };
  auto scalar_float_from_scalar =
      [](const Tensor &t) -> std::optional<FScalar> {
    if (!t.isScalar())
      return std::nullopt;
    const auto &s = t.scalar();
    if (s.dtype == Dtype::Float64)
      return FScalar{true, s.v.float64};
    if (s.dtype == Dtype::Float32)
      return FScalar{false, (double)s.v.float32};
    return std::nullopt;
  };
  auto scalar_float_from_raw0 = [](const Tensor &t) -> std::optional<FScalar> {
    if (!t.isRaw())
      return std::nullopt;
    const auto &rt = t.raw();
    if (!rt.shape.isScalar())
      return std::nullopt;
    if (rt.type == Dtype::Float64) {
      if (rt.raw.size() != sizeof(f64))
        throw std::runtime_error("vkcnn: Sub raw f64 size mismatch");
      f64 v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return FScalar{true, (double)v};
    }
    if (rt.type == Dtype::Float32) {
      if (rt.raw.size() != sizeof(f32))
        throw std::runtime_error("vkcnn: Sub raw f32 size mismatch");
      f32 v;
      std::memcpy(&v, rt.raw.data(), sizeof v);
      return FScalar{false, (double)v};
    }
    return std::nullopt;
  };
  auto as_i64_scalar = [&](const Tensor &t) -> std::optional<int64_t> {
    if (auto v = scalar_int64_from_scalar(t))
      return v;
    if (auto v = scalar_int64_from_raw0(t))
      return v;
    return std::nullopt;
  };
  auto as_f_scalar = [&](const Tensor &t) -> std::optional<FScalar> {
    if (auto v = scalar_float_from_scalar(t))
      return v;
    if (auto v = scalar_float_from_raw0(t))
      return v;
    return std::nullopt;
  };

  // --- shape math helpers (integers only) ----------------------------------
  auto sub_dim_dim = [&](const Dim &x, const Dim &y) -> Dim {
    if (x.isConst() && y.isConst()) {
      uint64_t a = x.value(), b = y.value();
      if (a < b)
        throw std::runtime_error(
            "vkcnn: Sub would produce negative shape dimension");
      return Dim::Const(a - b);
    }
    // symbolic: build Sym = x - y (both Sym or int64_t) via symGraph
    auto toSymOrI64 = [&](const Dim &d) -> std::variant<int64_t, Sym> {
      if (d.isConst()) {
        uint64_t v = d.value();
        if (v > (uint64_t)std::numeric_limits<int64_t>::max())
          throw std::runtime_error(
              "vkcnn: Dim constant too large for symbolic int64");
        return (int64_t)v;
      }
      return d.sym();
    };
    auto X = toSymOrI64(x);
    auto Y = toSymOrI64(y);
    Sym res = std::visit(
        [&](auto &&vx, auto &&vy) -> Sym {
          return state.symGraph->sub(vx,
                                     vy); // integer-only symbolic subtraction
        },
        X, Y);
    return Dim::Symbol(std::move(res));
  };
  auto sub_dim_i64 = [&](const Dim &x, int64_t y) -> Dim {
    if (x.isConst()) {
      uint64_t a = x.value();
      if (y < 0) {
        // a - (negative) => a + |y| (safe wrt underflow, might overflow 64-bit
        // but dims are usually small)
        uint64_t add = (uint64_t)(-y);
        uint64_t r = a + add;
        if (r < a)
          throw std::runtime_error("vkcnn: Sub overflow in shape math");
        return Dim::Const(r);
      } else {
        if (a < (uint64_t)y)
          throw std::runtime_error(
              "vkcnn: Sub would produce negative shape dimension");
        return Dim::Const(a - (uint64_t)y);
      }
    } else {
      // symbolic minus integer
      Sym res = state.symGraph->sub(x.sym(), y);
      return Dim::Symbol(std::move(res));
    }
  };

  auto broadcast_vec_vec = [&](const ShapeVector &X,
                               const ShapeVector &Y) -> ShapeVector {
    const size_t nx = X.size(), ny = Y.size();
    if (nx == ny) {
      ShapeVector out(nx);
      for (size_t i = 0; i < nx; ++i)
        out[i] = sub_dim_dim(X[i], Y[i]);
      return out;
    } else if (nx == 1 && ny >= 1) {
      ShapeVector out(ny);
      for (size_t i = 0; i < ny; ++i)
        out[i] = sub_dim_dim(X[0], Y[i]);
      return out;
    } else if (ny == 1 && nx >= 1) {
      ShapeVector out(nx);
      for (size_t i = 0; i < nx; ++i)
        out[i] = sub_dim_dim(X[i], Y[0]);
      return out;
    }
    throw std::runtime_error(
        "vkcnn: Sub ShapeTensor broadcast failed (lengths incompatible)");
  };

  // === Case A: Shape ⊗ Shape (integers only)
  if (A.isShape() && B.isShape()) {
    const auto &sa = A.shapeTensor();
    const auto &sb = B.shapeTensor();
    if (sa.isScalar() || sb.isScalar())
      throw std::runtime_error("vkcnn: Sub on scalar ShapeTensor is invalid");
    ShapeVector out = broadcast_vec_vec(sa.dims(), sb.dims()); // or .dims()
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // Try to lift scalars (int or float) from either side
  const auto Ai = as_i64_scalar(A);
  const auto Bi = as_i64_scalar(B);
  const auto Af = as_f_scalar(A);
  const auto Bf = as_f_scalar(B);

  // === Case B: Shape ⊗ Scalar(int)
  if (A.isShape() && Bi.has_value()) {
    const auto &sa = A.shapeTensor();
    if (sa.isScalar())
      throw std::runtime_error("vkcnn: Sub on scalar ShapeTensor is invalid");
    ShapeVector out(sa.dims().size());
    for (size_t i = 0; i < sa.dims().size(); ++i)
      out[i] = sub_dim_i64(sa.dims()[i], *Bi);
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }
  if (Ai.has_value() && B.isShape()) {
    const auto &sb = B.shapeTensor();
    if (sb.isScalar())
      throw std::runtime_error("vkcnn: Sub on scalar ShapeTensor is invalid");
    ShapeVector out(sb.dims().size());
    for (size_t i = 0; i < sb.dims().size(); ++i) {
      // (Ai - dim[i])
      out[i] = sub_dim_dim(Dim::Const((uint64_t)(*Ai < 0 ? -*Ai : *Ai)),
                           sb.dims()[i]);
    }
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // === Case C: scalar ⊗ scalar
  // If any side is float -> float math; else integer math.
  if ((Af.has_value() || Bf.has_value()) ||
      (Ai.has_value() && Bi.has_value())) {
    if (Af.has_value() || Bf.has_value()) {
      // float path (promote int to float if mixed)
      double av = Af ? Af->v : (Ai ? (double)*Ai : 0.0);
      double bv = Bf ? Bf->v : (Bi ? (double)*Bi : 0.0);
      bool out64 = (Af && Af->is64) || (Bf && Bf->is64);
      if (out64)
        return {Tensor::Scalar((f64)(av - bv))};
      else
        return {Tensor::Scalar((f32)(av - bv))};
    } else {
      // integer path
      return {Tensor::Scalar((int64_t)(*Ai - *Bi))};
    }
  }

  // === Case D: RawTensor (0D) ⊗ anything not lifted above -> unsupported for
  // now (Non-0D RawTensors not implemented; add if you want full
  // constant-folding.)

  if (A.isUnknown() || B.isUnknown()) {
    throw std::runtime_error("vkcnn: Sub on unknown input kind");
  }
  throw std::runtime_error(
      fmt::format("vkcnn: Sub not supported for given input kinds (node='{}')",
                  node.name()));
}

} // namespace vkcnn::details
