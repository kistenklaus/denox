#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_ConstantOfShape(
    [[maybe_unused]] ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Tensor> &attributes,
    opset_version /*version*/, [[maybe_unused]] const onnx::NodeProto &node) {

  // --- contract
  if (outputCount != 1) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape must produce exactly one output");
  }
  if (inputs.size() != 1 || !inputs[0].has_value()) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape requires exactly one input (shape)");
  }
  const Tensor &shape_in = *inputs[0];

  if (shape_in.isRuntimeTensor()) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape: runtime shape input is not supported");
  }
  if (shape_in.isString()) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape: string shape input is not supported");
  }

  // --- read the desired output shape as a vector<uint64_t> (non-negative,
  // static)
  std::vector<uint64_t> out_sizes;

  auto read_shape_from_shape_tensor = [&](const ShapeTensor &st) {
    if (st.isScalar()) {
      // Shape input must be 1-D vector (possibly length 0); scalar shape tensor
      // is invalid here.
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape input must be 1-D, not scalar");
    }
    for (const Dim &d : st.dims()) {
      if (!d.isConst()) {
        throw std::runtime_error(
            "vkcnn: ConstantOfShape: shape must be static (no symbolic dims)");
      }
      uint64_t v = d.value();
      // non-negative already guaranteed by uint64_t; reject negatives at source
      // readers below
      out_sizes.push_back(v);
    }
  };

  auto read_shape_from_raw_1d_int = [&](const RawTensor &rt) {
    // must be integer dtype, 1-D int64/int32/int16/…; allow empty vector (=>
    // scalar output)
    auto is_int_dtype = [](Dtype dt) -> bool {
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
    if (!is_int_dtype(rt.type)) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape tensor must be integer dtype");
    }
    if (rt.shape.isScalar()) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape input must be 1-D tensor");
    }
    const auto &sv = rt.shape.dims();
    if (sv.size() != 1) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape input must be 1-D");
    }
    if (!sv[0].isConst()) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape length must be static");
    }
    size_t n = static_cast<size_t>(sv[0].value());
    const size_t esz = dtype_size(rt.type);
    if (rt.raw.size() != n * esz) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: shape raw payload size mismatch");
    }

    auto read_i64_at = [&](size_t i) -> int64_t {
      switch (rt.type) {
      case Dtype::Int64:
        return reinterpret_cast<const int64_t *>(rt.raw.data())[i];
      case Dtype::Int32:
        return (int64_t)reinterpret_cast<const int32_t *>(rt.raw.data())[i];
      case Dtype::Int16:
        return (int64_t)reinterpret_cast<const int16_t *>(rt.raw.data())[i];
      case Dtype::Int8:
        return (int64_t)reinterpret_cast<const int8_t *>(rt.raw.data())[i];
      case Dtype::Uint64:
        return (int64_t)reinterpret_cast<const uint64_t *>(rt.raw.data())[i];
      case Dtype::Uint32:
        return (int64_t)reinterpret_cast<const uint32_t *>(rt.raw.data())[i];
      case Dtype::Uint16:
        return (int64_t)reinterpret_cast<const uint16_t *>(rt.raw.data())[i];
      case Dtype::Uint8:
        return (int64_t)reinterpret_cast<const uint8_t *>(rt.raw.data())[i];
      default:
        throw std::logic_error("vkcnn: internal shape read");
      }
    };

    out_sizes.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      int64_t v = read_i64_at(i);
      if (v < 0) {
        throw std::runtime_error(
            "vkcnn: ConstantOfShape: negative dimension is invalid");
      }
      out_sizes.push_back(static_cast<uint64_t>(v));
    }
  };

  auto read_shape_from_scalar_int = [&](const ScalarTensor &s) {
    // interpret a single integer scalar as a 1-D shape of length 1: [s]
    auto as_i64 = [&]() -> std::optional<int64_t> {
      switch (s.dtype) {
      case Dtype::Int64:
        return s.v.i;
      case Dtype::Int32:
      case Dtype::Int16:
      case Dtype::Int8:
        return (int64_t)s.v.i;
      case Dtype::Uint64:
        if (s.v.u > (uint64_t)std::numeric_limits<int64_t>::max())
          throw std::runtime_error(
              "vkcnn: ConstantOfShape: scalar shape too large");
        return (int64_t)s.v.u;
      case Dtype::Uint32:
      case Dtype::Uint16:
      case Dtype::Uint8:
        return (int64_t)s.v.u;
      default:
        return std::nullopt;
      }
    }();
    if (!as_i64.has_value()) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: scalar shape must be integer");
    }
    if (*as_i64 < 0) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: negative dimension is invalid");
    }
    out_sizes.push_back((uint64_t)*as_i64);
  };

  if (shape_in.isShape()) {
    read_shape_from_shape_tensor(shape_in.shapeTensor());
  } else if (shape_in.isRaw()) {
    read_shape_from_raw_1d_int(shape_in.raw());
  } else if (shape_in.isScalar()) {
    read_shape_from_scalar_int(shape_in.scalar());
  } else if (shape_in.isList()) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape: list shape input not supported");
  } else if (shape_in.isUnknown()) {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape: unknown shape input kind");
  } else {
    throw std::runtime_error(
        "vkcnn: ConstantOfShape: unsupported shape input kind");
  }

  // --- compute element count (with overflow check)
  auto safe_prod = [&](const std::vector<uint64_t> &dims) -> size_t {
    if (dims.empty())
      return (size_t)1; // scalar
    size_t acc = 1;
    for (uint64_t d : dims) {
      if (d == 0) {
        acc = 0;
        break;
      }
      if (acc > std::numeric_limits<size_t>::max() / (size_t)d)
        throw std::runtime_error(
            "vkcnn: ConstantOfShape: element count overflows size_t");
      acc *= (size_t)d;
    }
    return acc;
  };
  const size_t N = safe_prod(out_sizes);

  // --- parse attribute 'value' (optional): single-element numeric tensor
  Dtype out_dtype = Dtype::Float32; // default per ONNX
  std::vector<std::byte> elem_bytes;

  auto load_value_from_scalar = [&](const ScalarTensor &s) {
    switch (s.dtype) {
    case Dtype::Float64: {
      out_dtype = Dtype::Float64;
      elem_bytes.resize(dtype_size(out_dtype));
      std::memcpy(elem_bytes.data(), &s.v.float64, sizeof(f64));
      break;
    }
    case Dtype::Float32: {
      out_dtype = Dtype::Float32;
      elem_bytes.resize(dtype_size(out_dtype));
      std::memcpy(elem_bytes.data(), &s.v.float32, sizeof(f32));
      break;
    }
    case Dtype::Int64:
    case Dtype::Int32:
    case Dtype::Int16:
    case Dtype::Int8:
    case Dtype::Uint64:
    case Dtype::Uint32:
    case Dtype::Uint16:
    case Dtype::Uint8: {
      out_dtype = s.dtype;
      elem_bytes.resize(dtype_size(out_dtype));
      // write the exact underlying integer width
      switch (s.dtype) {
      case Dtype::Int64:
        std::memcpy(elem_bytes.data(), &s.v.i, sizeof(std::int64_t));
        break;
      case Dtype::Int32:
        std::memcpy(elem_bytes.data(), &s.v.i, sizeof(std::int32_t));
        break;
      case Dtype::Int16:
        std::memcpy(elem_bytes.data(), &s.v.i, sizeof(std::int16_t));
        break;
      case Dtype::Int8:
        std::memcpy(elem_bytes.data(), &s.v.i, sizeof(std::int8_t));
        break;
      case Dtype::Uint64:
        std::memcpy(elem_bytes.data(), &s.v.u, sizeof(std::uint64_t));
        break;
      case Dtype::Uint32:
        std::memcpy(elem_bytes.data(), &s.v.u, sizeof(std::uint32_t));
        break;
      case Dtype::Uint16:
        std::memcpy(elem_bytes.data(), &s.v.u, sizeof(std::uint16_t));
        break;
      case Dtype::Uint8:
        std::memcpy(elem_bytes.data(), &s.v.u, sizeof(std::uint8_t));
        break;
      default:
        break;
      }
      break;
    }
    default:
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: value scalar dtype not supported");
    }
  };

  auto load_value_from_raw1 = [&](const RawTensor &rt) {
    // one element total
    // allow integer and f32/f64; reject string/others
    auto is_allowed = [&](Dtype dt) -> bool {
      switch (dt) {
      case Dtype::Int8:
      case Dtype::Int16:
      case Dtype::Int32:
      case Dtype::Int64:
      case Dtype::Uint8:
      case Dtype::Uint16:
      case Dtype::Uint32:
      case Dtype::Uint64:
      case Dtype::Float32:
      case Dtype::Float64:
        return true;
      default:
        return false;
      }
    };
    if (!is_allowed(rt.type)) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: value attr dtype not supported");
    }
    // element-count must be 1
    size_t count = 1;
    if (rt.shape.isTensor()) {
      // compute product of dims
      count = 1;
      for (const Dim &d : rt.shape.dims()) {
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: ConstantOfShape: value shape must be static");
        count *= (size_t)d.value();
      }
    }
    if (count != 1) {
      throw std::runtime_error("vkcnn: ConstantOfShape: value attribute must "
                               "contain exactly one element");
    }
    const size_t esz = dtype_size(rt.type);
    if (rt.raw.size() != esz) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: value raw payload size mismatch");
    }
    out_dtype = rt.type;
    elem_bytes = rt.raw; // copy
  };

  auto it_val = attributes.find("value");
  if (it_val == attributes.end() || it_val->second.isUnknown()) {
    // default: float32(0)
    out_dtype = Dtype::Float32;
    elem_bytes.assign(dtype_size(out_dtype), std::byte{0});
  } else {
    const Tensor &val = it_val->second;
    if (val.isRuntimeTensor()) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: runtime 'value' is not supported");
    }
    if (val.isString()) {
      throw std::runtime_error(
          "vkcnn: ConstantOfShape: string 'value' is not supported");
    }
    if (val.isScalar()) {
      load_value_from_scalar(val.scalar());
    } else if (val.isRaw()) {
      load_value_from_raw1(val.raw());
    } else {
      throw std::runtime_error("vkcnn: ConstantOfShape: 'value' must be a "
                               "scalar or 1-element constant tensor");
    }
  }

  // --- build output RawTensor
  RawTensor out;
  out.type = out_dtype;

  if (out_sizes.empty()) {
    out.shape = ShapeTensor::Scalar(); // scalar output
  } else {
    ShapeVector dims;
    dims.reserve(out_sizes.size());
    for (uint64_t v : out_sizes)
      dims.push_back(Dim::Const(v));
    out.shape = ShapeTensor::Tensor(std::move(dims));
  }

  const size_t esz = dtype_size(out_dtype);
  out.raw.resize(N * esz);

  if (N == 0) {
    // nothing to fill; shape is valid with zero elements
    return {Tensor::Raw(std::move(out))};
  }

  // Fill all elements with the single value
  for (size_t i = 0; i < N; ++i) {
    std::memcpy(out.raw.data() + i * esz, elem_bytes.data(), esz);
  }

  return {Tensor::Raw(std::move(out))};
}

} // namespace vkcnn::details
