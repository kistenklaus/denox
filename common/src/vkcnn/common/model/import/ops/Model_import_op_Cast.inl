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
import_op_Cast(ImportState &state,
               std::span<const std::optional<Tensor>> inputs,
               std::size_t outputCount,
               const std::unordered_map<std::string, Tensor> &attributes,
               opset_version /*version*/, const onnx::NodeProto &node) {

  // --- contract
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Cast must produce exactly one output");
  }
  if (inputs.size() != 1 || !inputs[0].has_value()) {
    throw std::runtime_error("vkcnn: Cast requires exactly one input");
  }
  const Tensor &X = *inputs[0];

  if (X.isRuntimeTensor()) {
    throw std::runtime_error("vkcnn: Cast on runtime tensors is not supported");
  }
  if (X.isString()) {
    throw std::runtime_error("vkcnn: Cast on string tensors is not supported");
  }

  // --- parse attribute 'to' (ONNX TensorProto_DataType)
  auto get_int_scalar = [](const Tensor &t) -> std::int64_t {
    if (!t.isScalar())
      throw std::runtime_error(
          "vkcnn: Cast attribute 'to' must be an integer scalar");
    const auto &s = t.scalar();
    if (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
        s.dtype == Dtype::Int16)
      return (std::int64_t)s.v.i;
    if (s.dtype == Dtype::Uint64 || s.dtype == Dtype::Uint32 ||
        s.dtype == Dtype::Uint16 || s.dtype == Dtype::Uint8)
      return (std::int64_t)s.v.u;
    throw std::runtime_error(
        "vkcnn: Cast attribute 'to' must be an integer scalar");
  };

  auto it = attributes.find("to");
  if (it == attributes.end()) {
    throw std::runtime_error("vkcnn: Cast requires attribute 'to'");
  }
  const std::int64_t onnx_to = get_int_scalar(it->second);

  // Map ONNX TensorProto_DataType -> your Dtype
  auto map_to_dtype = [&](std::int64_t code) -> Dtype {
    switch (code) {
    case 1:
      return Dtype::Float32; // FLOAT
    case 2:
      return Dtype::Uint8;
    case 3:
      return Dtype::Int8;
    case 4:
      return Dtype::Uint16;
    case 5:
      return Dtype::Int16;
    case 6:
      return Dtype::Int32;
    case 7:
      return Dtype::Int64;
    case 8:
      throw std::runtime_error("vkcnn: Cast to string is not supported");
    case 9:
      throw std::runtime_error("vkcnn: Cast to bool is not supported");
    case 10:
      throw std::runtime_error("vkcnn: Cast to float16 is not supported yet");
    case 11:
      return Dtype::Float64; // DOUBLE
    case 12:
      return Dtype::Uint32;
    case 13:
      return Dtype::Uint64;
    // 14,15 complex; 16 bf16; 17/18 float8; newer codes exist
    default:
      throw std::runtime_error(
          "vkcnn: Cast target dtype not supported by importer");
    }
  };
  const Dtype target = map_to_dtype(onnx_to);

  auto is_int_dt = [](Dtype dt) -> bool {
    switch (dt) {
    case Dtype::Int8:
    case Dtype::Int16:
    case Dtype::Int32:
    case Dtype::Int64:
    case Dtype::Uint8:
    case Dtype::Uint16:
    case Dtype::Uint32:
    case Dtype::Uint64:
      return true;
    default:
      return false;
    }
  };
  auto is_float_dt = [](Dtype dt) -> bool {
    return dt == Dtype::Float32 || dt == Dtype::Float64;
  };

  // -------------------- ShapeTensor: no-op for integer targets
  // --------------------
  if (X.isShape()) {
    if (!is_int_dt(target)) {
      // We cannot materialize symbolic dims as floats; reject to keep semantics
      // sound.
      throw std::runtime_error(
          "vkcnn: Cast on ShapeTensor only supports integer targets");
    }
    // No-op: dims remain (Const/Sym). We keep the ShapeTensor kind.
    return {X};
  }

  // -------------------- ScalarTensor casting --------------------
  if (X.isScalar()) {
    const auto &s = X.scalar();

    auto emit_int_scalar = [&](long long v, Dtype dst) -> Tensor {
      switch (dst) {
      case Dtype::Int64:
        return Tensor::Scalar((std::int64_t)v);
      case Dtype::Int32:
        return Tensor::Scalar((std::int32_t)v);
      case Dtype::Int16:
        return Tensor::Scalar((std::int16_t)v);
      case Dtype::Int8:
        return Tensor::Scalar((std::int8_t)v);
      case Dtype::Uint64:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        return Tensor::Scalar((std::uint64_t)v);
      case Dtype::Uint32:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        return Tensor::Scalar((std::uint32_t)v);
      case Dtype::Uint16:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        return Tensor::Scalar((std::uint16_t)v);
      case Dtype::Uint8:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        return Tensor::Scalar((std::uint8_t)v);
      default:
        throw std::logic_error("vkcnn: internal int emit");
      }
    };
    auto emit_float_scalar = [&](double v, Dtype dst) -> Tensor {
      if (!std::isfinite(v))
        throw std::runtime_error("vkcnn: Cast of NaN/Inf not supported");
      if (dst == Dtype::Float64)
        return Tensor::Scalar((f64)v);
      if (dst == Dtype::Float32)
        return Tensor::Scalar((f32)v);
      throw std::logic_error("vkcnn: internal float emit");
    };

    // source kind
    if (is_int_dt(target)) {
      // cast to integer
      switch (s.dtype) {
      case Dtype::Int64:
        return {emit_int_scalar(s.v.i, target)};
      case Dtype::Int32:
      case Dtype::Int16:
      case Dtype::Int8:
        return {emit_int_scalar((long long)s.v.i, target)};
      case Dtype::Uint64:
        if (s.v.u > (std::uint64_t)std::numeric_limits<long long>::max())
          throw std::runtime_error("vkcnn: Cast overflow");
        return {emit_int_scalar((long long)s.v.u, target)};
      case Dtype::Uint32:
      case Dtype::Uint16:
      case Dtype::Uint8:
        return {emit_int_scalar((long long)s.v.u, target)};
      case Dtype::Float64:
        return {emit_int_scalar((long long)s.v.float64,
                               target)}; // trunc toward 0
      case Dtype::Float32:
        return {emit_int_scalar((long long)s.v.float32, target)};
      default:
        throw std::runtime_error(
            "vkcnn: Cast scalar: unsupported source dtype");
      }
    } else if (is_float_dt(target)) {
      // cast to float
      switch (s.dtype) {
      case Dtype::Float64:
        return {emit_float_scalar((double)s.v.float64, target)};
      case Dtype::Float32:
        return {emit_float_scalar((double)s.v.float32, target)};
      case Dtype::Int64:
        return {emit_float_scalar((double)s.v.i, target)};
      case Dtype::Int32:
      case Dtype::Int16:
      case Dtype::Int8:
        return {emit_float_scalar((double)s.v.i, target)};
      case Dtype::Uint64:
        return {emit_float_scalar((double)s.v.u, target)};
      case Dtype::Uint32:
      case Dtype::Uint16:
      case Dtype::Uint8:
        return {emit_float_scalar((double)s.v.u, target)};
      default:
        throw std::runtime_error(
            "vkcnn: Cast scalar: unsupported source dtype");
      }
    } else {
      throw std::runtime_error("vkcnn: Cast scalar: unsupported target dtype");
    }
  }

  // -------------------- RawTensor casting (constants) --------------------
  if (X.isRaw()) {
    const RawTensor &rt = X.raw();
    if (rt.type == Dtype::String)
      throw std::runtime_error(
          "vkcnn: Cast on RawTensor<string> is not supported");

    // element count
    auto elem_count = [](const ShapeTensor &sh) -> std::size_t {
      if (sh.isScalar())
        return 1;
      std::size_t n = 1;
      for (const Dim &d : sh.dims()) {
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: Cast RawTensor requires static shape");
        n *= static_cast<std::size_t>(d.value());
      }
      return n;
    };
    const std::size_t N = elem_count(rt.shape);

    // source readers
    auto read_i64 = [&](std::size_t i) -> long long {
      switch (rt.type) {
      case Dtype::Int64:
        return reinterpret_cast<const std::int64_t *>(rt.raw.data())[i];
      case Dtype::Int32:
        return (long long)reinterpret_cast<const std::int32_t *>(
            rt.raw.data())[i];
      case Dtype::Int16:
        return (long long)reinterpret_cast<const std::int16_t *>(
            rt.raw.data())[i];
      case Dtype::Int8:
        return (long long)reinterpret_cast<const std::int8_t *>(
            rt.raw.data())[i];
      case Dtype::Uint64: {
        std::uint64_t u =
            reinterpret_cast<const std::uint64_t *>(rt.raw.data())[i];
        if (u > (std::uint64_t)std::numeric_limits<long long>::max())
          throw std::runtime_error("vkcnn: Cast overflow (u64->int)");
        return (long long)u;
      }
      case Dtype::Uint32:
        return (long long)reinterpret_cast<const std::uint32_t *>(
            rt.raw.data())[i];
      case Dtype::Uint16:
        return (long long)reinterpret_cast<const std::uint16_t *>(
            rt.raw.data())[i];
      case Dtype::Uint8:
        return (long long)reinterpret_cast<const std::uint8_t *>(
            rt.raw.data())[i];
      case Dtype::Float64:
        return (long long)reinterpret_cast<const f64 *>(
            rt.raw.data())[i]; // trunc toward 0
      case Dtype::Float32:
        return (long long)reinterpret_cast<const f32 *>(rt.raw.data())[i];
      default:
        throw std::runtime_error(
            "vkcnn: Cast RawTensor: unsupported source dtype");
      }
    };
    auto read_f64 = [&](std::size_t i) -> double {
      switch (rt.type) {
      case Dtype::Float64:
        return (double)reinterpret_cast<const f64 *>(rt.raw.data())[i];
      case Dtype::Float32:
        return (double)reinterpret_cast<const f32 *>(rt.raw.data())[i];
      case Dtype::Int64:
        return (double)reinterpret_cast<const std::int64_t *>(rt.raw.data())[i];
      case Dtype::Int32:
        return (double)reinterpret_cast<const std::int32_t *>(rt.raw.data())[i];
      case Dtype::Int16:
        return (double)reinterpret_cast<const std::int16_t *>(rt.raw.data())[i];
      case Dtype::Int8:
        return (double)reinterpret_cast<const std::int8_t *>(rt.raw.data())[i];
      case Dtype::Uint64:
        return (double)reinterpret_cast<const std::uint64_t *>(
            rt.raw.data())[i];
      case Dtype::Uint32:
        return (double)reinterpret_cast<const std::uint32_t *>(
            rt.raw.data())[i];
      case Dtype::Uint16:
        return (double)reinterpret_cast<const std::uint16_t *>(
            rt.raw.data())[i];
      case Dtype::Uint8:
        return (double)reinterpret_cast<const std::uint8_t *>(rt.raw.data())[i];
      default:
        throw std::runtime_error(
            "vkcnn: Cast RawTensor: unsupported source dtype");
      }
    };

    RawTensor out;
    out.type = target;
    out.shape = rt.shape; // shape preserved
    out.raw.resize(N * dtype_size(target));

    // Write
    auto write_int_at = [&](std::size_t i, long long v) {
      switch (target) {
      case Dtype::Int64:
        reinterpret_cast<std::int64_t *>(out.raw.data())[i] = (std::int64_t)v;
        break;
      case Dtype::Int32:
        reinterpret_cast<std::int32_t *>(out.raw.data())[i] = (std::int32_t)v;
        break;
      case Dtype::Int16:
        reinterpret_cast<std::int16_t *>(out.raw.data())[i] = (std::int16_t)v;
        break;
      case Dtype::Int8:
        reinterpret_cast<std::int8_t *>(out.raw.data())[i] = (std::int8_t)v;
        break;
      case Dtype::Uint64:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        reinterpret_cast<std::uint64_t *>(out.raw.data())[i] = (std::uint64_t)v;
        break;
      case Dtype::Uint32:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        reinterpret_cast<std::uint32_t *>(out.raw.data())[i] = (std::uint32_t)v;
        break;
      case Dtype::Uint16:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        reinterpret_cast<std::uint16_t *>(out.raw.data())[i] = (std::uint16_t)v;
        break;
      case Dtype::Uint8:
        if (v < 0)
          throw std::runtime_error("vkcnn: Cast negative to unsigned");
        reinterpret_cast<std::uint8_t *>(out.raw.data())[i] = (std::uint8_t)v;
        break;
      default:
        throw std::logic_error("vkcnn: internal write_int_at");
      }
    };
    auto write_float_at = [&](std::size_t i, double v) {
      if (!std::isfinite(v))
        throw std::runtime_error("vkcnn: Cast of NaN/Inf not supported");
      if (target == Dtype::Float64)
        reinterpret_cast<f64 *>(out.raw.data())[i] = (f64)v;
      else if (target == Dtype::Float32)
        reinterpret_cast<f32 *>(out.raw.data())[i] = (f32)v;
      else
        throw std::logic_error("vkcnn: internal write_float_at");
    };

    if (is_int_dt(target)) {
      for (std::size_t i = 0; i < N; ++i) {
        long long v = read_i64(i);
        write_int_at(i, v);
      }
    } else if (is_float_dt(target)) {
      for (std::size_t i = 0; i < N; ++i) {
        double v = read_f64(i);
        write_float_at(i, v);
      }
    } else {
      throw std::runtime_error(
          "vkcnn: Cast RawTensor: unsupported target dtype");
    }

    return {Tensor::Raw(std::move(out))};
  }

  // -------------------- Unknown / unsupported kind --------------------
  if (X.isUnknown()) {
    throw std::runtime_error("vkcnn: Cast on unknown input kind");
  }
  throw std::runtime_error(
      fmt::format("vkcnn: Cast not supported for given input kind (node='{}')",
                  node.name()));
}

} // namespace vkcnn::details
