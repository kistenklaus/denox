#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/base.h>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Mod(ImportState &state, std::span<const std::optional<Tensor>> inputs,
              std::size_t outputCount,
              const std::unordered_map<std::string, Tensor> &attributes,
              opset_version /*version*/, const onnx::NodeProto &node) {

  // --- contract
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Mod requires exactly two inputs. Got {} (node='{}')",
        inputs.size(), node.op_type()));
  }
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Mod must produce exactly one output");
  }
  const Tensor &A = *inputs[0];
  const Tensor &B = *inputs[1];

  // --- reject unsupported kinds early
  auto is_float_scalar = [](const Tensor &t) -> bool {
    if (!t.isScalar())
      return false;
    const auto &s = t.scalar();
    return s.dtype == Dtype::Float16 || s.dtype == Dtype::Float32 ||
           s.dtype == Dtype::Float64;
  };
  auto is_raw_float_scalar = [](const Tensor &t) -> bool {
    if (!t.isRaw())
      return false;
    const auto &rt = t.raw();
    if (!rt.shape.isScalar())
      return false;
    return rt.type == Dtype::Float16 || rt.type == Dtype::Float32 ||
           rt.type == Dtype::Float64;
  };
  auto is_raw_string_scalar = [](const Tensor &t) -> bool {
    if (!t.isRaw())
      return false;
    const auto &rt = t.raw();
    return rt.shape.isScalar() && rt.type == Dtype::String;
  };

  if (A.isRuntimeTensor() || B.isRuntimeTensor()) {
    throw std::runtime_error("vkcnn: Mod on runtime tensors is not supported");
  }
  if (A.isString() || B.isString() || is_raw_string_scalar(A) ||
      is_raw_string_scalar(B)) {
    throw std::runtime_error("vkcnn: Mod on string tensors is not supported");
  }
  if (is_float_scalar(A) || is_float_scalar(B) || is_raw_float_scalar(A) ||
      is_raw_float_scalar(B)) {
    throw std::runtime_error("vkcnn: Mod for floating types is not supported");
  }

  // --- fmod attribute handling (reject if truthy)
  if (auto it = attributes.find("fmod"); it != attributes.end()) {
    const Tensor &fm = it->second;
    bool truthy = false;
    if (fm.isScalar()) {
      const auto &s = fm.scalar();
      if (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
          s.dtype == Dtype::Int16 || s.dtype == Dtype::Uint64 ||
          s.dtype == Dtype::Uint32 || s.dtype == Dtype::Uint16 ||
          s.dtype == Dtype::Uint8) {
        truthy = (s.v.u != 0) || (s.v.i != 0);
      }
    } else if (!fm.isUnknown()) {
      throw std::runtime_error("vkcnn: Mod attribute 'fmod' must be a scalar");
    }
    if (truthy) {
      throw std::runtime_error("vkcnn: Mod with attribute 'fmod'=1 (floating "
                               "remainder) is not supported");
    }
  }

  // --- helpers
  auto scalar_to_i64 = [](const Tensor &t) -> std::optional<int64_t> {
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
            "vkcnn: Mod scalar uint64 too large to fit int64");
      return static_cast<int64_t>(s.v.u);
    case Dtype::Uint32:
    case Dtype::Uint16:
    case Dtype::Uint8:
      return static_cast<int64_t>(s.v.u);
    default:
      return std::nullopt;
    }
  };

  auto raw_scalar_to_i64 = [](const Tensor &t) -> std::optional<int64_t> {
    if (!t.isRaw())
      return std::nullopt;
    const auto &rt = t.raw();
    if (!rt.shape.isScalar())
      return std::nullopt;
    auto need = [&](size_t n) {
      if (rt.raw.size() != n)
        throw std::runtime_error("vkcnn: Mod raw scalar payload size mismatch");
    };
    switch (rt.type) {
    case Dtype::Int64: {
      need(sizeof(int64_t));
      int64_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return v;
    }
    case Dtype::Int32: {
      need(sizeof(int32_t));
      int32_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    case Dtype::Int16: {
      need(sizeof(int16_t));
      int16_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    case Dtype::Int8: {
      need(sizeof(int8_t));
      int8_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    case Dtype::Uint64: {
      need(sizeof(uint64_t));
      uint64_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error(
            "vkcnn: Mod raw uint64 scalar too large to fit int64");
      return static_cast<int64_t>(v);
    }
    case Dtype::Uint32: {
      need(sizeof(uint32_t));
      uint32_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    case Dtype::Uint16: {
      need(sizeof(uint16_t));
      uint16_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    case Dtype::Uint8: {
      need(sizeof(uint8_t));
      uint8_t v;
      std::memcpy(&v, rt.raw.data(), sizeof(v));
      return static_cast<int64_t>(v);
    }
    default:
      return std::nullopt; // not an integer scalar
    }
  };

  auto as_i64_scalar = [&](const Tensor &t) -> std::optional<int64_t> {
    if (auto v = scalar_to_i64(t))
      return v;
    if (auto v = raw_scalar_to_i64(t))
      return v;
    return std::nullopt;
  };

  auto mod_dim_dim = [&](const Dim &x, const Dim &y) -> Dim {
    if (y.isConst() && y.value() == 0ull) {
      throw std::runtime_error("vkcnn: Mod by zero");
    }
    if (x.isConst() && y.isConst()) {
      uint64_t a = x.value();
      uint64_t b = y.value();
      return Dim::Const(a % b);
    }
    auto toSymOrI64 = [&](const Dim &d) -> std::variant<int64_t, Sym> {
      if (d.isConst()) {
        uint64_t v = d.value();
        if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          throw std::runtime_error(
              "vkcnn: Dim constant too large for symbolic int64");
        return static_cast<int64_t>(v);
      }
      return d.sym();
    };
    auto X = toSymOrI64(x);
    auto Y = toSymOrI64(y);
    Sym res = std::visit(
        [&](auto &&vx, auto &&vy) -> Sym {
          return state.symGraph->mod(vx, vy);
        },
        X, Y);
    return Dim::Symbol(std::move(res));
  };

  auto mod_dim_i64 = [&](const Dim &x, int64_t y) -> Dim {
    if (y == 0)
      throw std::runtime_error("vkcnn: Mod by zero");
    if (x.isConst()) {
      uint64_t a = x.value();
      uint64_t b = static_cast<uint64_t>(y < 0 ? -y : y);
      if (b == 0ull)
        throw std::runtime_error("vkcnn: Mod by zero");
      return Dim::Const(a % b);
    }
    Sym res = state.symGraph->mod(x.sym(), y);
    return Dim::Symbol(std::move(res));
  };

  auto i64_mod_i64 = [&](int64_t a, int64_t b) -> int64_t {
    if (b == 0)
      throw std::runtime_error("vkcnn: Mod by zero");
    return a % b; // integer remainder (fmod=0)
  };

  auto broadcast_vec_vec = [&](const ShapeVector &X,
                               const ShapeVector &Y) -> ShapeVector {
    const size_t nx = X.size(), ny = Y.size();
    if (nx == ny) {
      ShapeVector out(nx);
      for (size_t i = 0; i < nx; ++i)
        out[i] = mod_dim_dim(X[i], Y[i]);
      return out;
    } else if (nx == 1 && ny >= 1) {
      ShapeVector out(ny);
      for (size_t i = 0; i < ny; ++i)
        out[i] = mod_dim_dim(X[0], Y[i]);
      return out;
    } else if (ny == 1 && nx >= 1) {
      ShapeVector out(nx);
      for (size_t i = 0; i < nx; ++i)
        out[i] = mod_dim_dim(X[i], Y[0]);
      return out;
    }
    throw std::runtime_error(
        "vkcnn: Mod ShapeTensor broadcast failed (lengths incompatible)");
  };

  // === CASE 1: Shape ⊗ Shape
  if (A.isShape() && B.isShape()) {
    const auto &sa = A.shapeTensor();
    const auto &sb = B.shapeTensor();
    if (sa.isScalar() || sb.isScalar())
      throw std::runtime_error("vkcnn: Mod on scalar ShapeTensor is invalid");
    ShapeVector out = broadcast_vec_vec(sa.dims(), sb.dims());
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // Try to lift scalars from either ScalarTensor or RawTensor(0D)
  const auto A_i64 = as_i64_scalar(A);
  const auto B_i64 = as_i64_scalar(B);

  // === CASE 2: Shape ⊗ Scalar
  if (A.isShape() && B_i64.has_value()) {
    const auto &sa = A.shapeTensor();
    if (sa.isScalar())
      throw std::runtime_error("vkcnn: Mod on scalar ShapeTensor is invalid");
    const int64_t rhs = *B_i64;
    ShapeVector out(sa.dims().size());
    for (size_t i = 0; i < sa.dims().size(); ++i)
      out[i] = mod_dim_i64(sa.dims()[i], rhs);
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }
  if (A_i64.has_value() && B.isShape()) {
    const auto &sb = B.shapeTensor();
    if (sb.isScalar())
      throw std::runtime_error("vkcnn: Mod on scalar ShapeTensor is invalid");
    const int64_t lhs = *A_i64;
    ShapeVector out(sb.dims().size());
    for (size_t i = 0; i < sb.dims().size(); ++i) {
      out[i] =
          mod_dim_dim(Dim::Const(static_cast<uint64_t>(lhs < 0 ? -lhs : lhs)),
                      sb.dims()[i]);
    }
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // === CASE 3: scalar ⊗ scalar (from ScalarTensor or RawTensor(0D))
  if (A_i64.has_value() && B_i64.has_value()) {
    return {Tensor::Scalar(i64_mod_i64(*A_i64, *B_i64))};
  }

  // (Optional) Non-scalar RawTensor support could go here later.

  if (A.isUnknown() || B.isUnknown()) {
    throw std::runtime_error("vkcnn: Mod on unknown input kind");
  }
  throw std::runtime_error(
      fmt::format("vkcnn: Mod not supported for given input kinds (node='{}')",
                  node.name()));
}

} // namespace vkcnn::details
