#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Reshape(
    [[maybe_unused]] ImportState &state,
    std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Tensor> &attributes,
    opset_version /*version*/, [[maybe_unused]] const onnx::NodeProto &node) {

  // --- contract
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Reshape must produce exactly one output");
  }
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value()) {
    throw std::runtime_error("vkcnn: Reshape requires 2 inputs: data and shape");
  }
  const Tensor &data  = *inputs[0];
  const Tensor &shape = *inputs[1];

  // --- rejects
  if (data.isRuntimeTensor() || shape.isRuntimeTensor())
    throw std::runtime_error("vkcnn: Reshape on runtime tensors is not supported");
  if (data.isString() || shape.isString())
    throw std::runtime_error("vkcnn: Reshape on string tensors is not supported");
  if (data.isList() || shape.isList())
    throw std::runtime_error("vkcnn: Reshape: list inputs not supported");
  if (data.isUnknown() || shape.isUnknown())
    throw std::runtime_error("vkcnn: Reshape: unknown input kind");

  // --- allowzero (default 0)
  auto get_allowzero = [&]() -> bool {
    auto it = attributes.find("allowzero");
    if (it == attributes.end() || it->second.isUnknown()) return false;
    const auto &a = it->second.scalar();
    switch (a.dtype) {
      case Dtype::Int64: case Dtype::Int32: case Dtype::Int16: return a.v.i != 0;
      case Dtype::Uint64: case Dtype::Uint32: case Dtype::Uint16: case Dtype::Uint8: return a.v.u != 0;
      default: throw std::runtime_error("vkcnn: Reshape attribute 'allowzero' must be integer");
    }
  };
  const bool allowzero = get_allowzero();

  // --- helpers
  auto static_elem_count_from_shape = [&](const ShapeTensor &st) -> std::pair<uint64_t,bool> {
    if (st.isScalar()) return {1, true};
    uint64_t n = 1;
    for (const Dim &d : st.dims()) {
      if (!d.isConst()) return {0,false};
      n *= d.value();
    }
    return {n,true};
  };

  auto data_shape_dims = [&]() -> ShapeVector {
    if (data.isScalar())          return ShapeVector{}; // rank-0
    if (data.isRaw()) {
      if (data.raw().shape.isScalar()) return ShapeVector{};
      return data.raw().shape.dims();
    }
    if (data.isShape()) {
      // Treat ShapeTensor AS DATA: a flat 1-D vector of Dim. Its "shape" is [len].
      // For element-count checks we only need len.
      ShapeVector v;
      if (!data.shapeTensor().isScalar()) v = data.shapeTensor().dims();
      return v;
    }
    throw std::runtime_error("vkcnn: Reshape: unsupported data kind");
  }();

  auto data_elem_count_known = [&]() -> std::pair<uint64_t,bool> {
    if (data.isScalar())  return {1,true};
    if (data.isRaw()) {
      const auto &rt = data.raw();
      if (rt.shape.isScalar()) return {1,true};
      uint64_t n=1;
      for (const Dim &d : rt.shape.dims()) {
        if (!d.isConst()) return {0,false};
        n *= d.value();
      }
      return {n,true};
    }
    if (data.isShape()) {
      // number of "elements" is number of Dims in the vector
      if (data.shapeTensor().isScalar()) return {0,true}; // empty vector
      return { (uint64_t)data.shapeTensor().dims().size(), true };
    }
    return {0,false};
  }();
  const uint64_t Nin = data_elem_count_known.first;
  const bool Nin_known = data_elem_count_known.second;

  if (data.isRaw()) {
    // sanity: raw size matches elem-count
    const auto &rt = data.raw();
    const size_t esz = dtype_size(rt.type);
    if (esz == 0) throw std::runtime_error("vkcnn: Reshape unsupported dtype");
    // If shape has symbolic dims we already would have flagged Nin_known=false above, but
    // we don't reach this branch unless all dims are const (see elem_count computation).
    if (rt.raw.size() != (size_t)Nin * esz)
      throw std::runtime_error("vkcnn: Reshape: input raw payload size does not match element count");
  }

  // Parse the SHAPE SPEC (input[1]) into a list of "requested dims" (may contain 0/-1 for int tensors).
  bool used_infer_neg1 = false;
  int64_t infer_index = -1;

  auto parse_shape_from_shape_tensor = [&](const ShapeTensor &st) -> ShapeVector {
    if (st.isScalar())
      throw std::runtime_error("vkcnn: Reshape: shape input must be a 1-D vector");
    const auto &spec = st.dims();
    const int64_t Rin = (int64_t)data_shape_dims.size(); // input rank (for zero-copy semantics)
    ShapeVector out; out.reserve(spec.size());
    for (size_t i = 0; i < spec.size(); ++i) {
      const Dim &s = spec[i];
      if (s.isConst()) {
        uint64_t v = s.value(); // non-negative only (ShapeTensor cannot encode negatives)
        if (v == 0ull && !allowzero) {
          if ((int64_t)i >= Rin) {
            throw std::runtime_error("vkcnn: Reshape: zero-dim copies from out-of-range input axis");
          }
          out.push_back(data_shape_dims[i]); // copy input dim at same index
        } else {
          out.push_back(Dim::Const(v));
        }
      } else {
        out.push_back(s); // keep symbol
      }
    }
    return out;
  };

  auto parse_shape_from_raw_1d_int = [&](const RawTensor &rt) -> ShapeVector {
    auto is_int_dt = [](Dtype dt)->bool {
      switch (dt) {
        case Dtype::Int8: case Dtype::Int16: case Dtype::Int32: case Dtype::Int64:
        case Dtype::Uint8: case Dtype::Uint16: case Dtype::Uint32: case Dtype::Uint64:
          return true;
        default: return false;
      }
    };
    if (!is_int_dt(rt.type)) throw std::runtime_error("vkcnn: Reshape: shape tensor must be integer dtype");
    if (!rt.shape.isScalar()) {
      const auto &sv = rt.shape.dims();
      if (sv.size() != 1) throw std::runtime_error("vkcnn: Reshape: shape input must be 1-D");
      if (!sv[0].isConst()) throw std::runtime_error("vkcnn: Reshape: shape length must be static");
    }
    size_t n = rt.shape.isScalar() ? 1 : (size_t)rt.shape.dims()[0].value();
    const size_t esz = dtype_size(rt.type);
    if (rt.raw.size() != n * esz) throw std::runtime_error("vkcnn: Reshape: shape raw payload size mismatch");

    ShapeVector out; out.reserve(n);
    const uint8_t *p = (const uint8_t*)rt.raw.data();
    auto read_i64 = [&](size_t i)->int64_t {
      switch (rt.type) {
        case Dtype::Int64:  return ((const int64_t*)p)[i];
        case Dtype::Int32:  return (int64_t)((const int32_t*)p)[i];
        case Dtype::Int16:  return (int64_t)((const int16_t*)p)[i];
        case Dtype::Int8:   return (int64_t)((const int8_t *)p)[i];
        case Dtype::Uint64: return (int64_t)((const uint64_t*)p)[i];
        case Dtype::Uint32: return (int64_t)((const uint32_t*)p)[i];
        case Dtype::Uint16: return (int64_t)((const uint16_t*)p)[i];
        case Dtype::Uint8:  return (int64_t)((const uint8_t *)p)[i];
        default: throw std::logic_error("vkcnn: internal");
      }
    };
    const int64_t Rin = (int64_t)data_shape_dims.size();
    for (size_t i = 0; i < n; ++i) {
      int64_t v = read_i64(i);
      if (v == -1) {
        if (used_infer_neg1) throw std::runtime_error("vkcnn: Reshape: at most one -1 is allowed");
        used_infer_neg1 = true; infer_index = (int64_t)i;
        out.push_back(Dim::Const(1)); // placeholder
      } else if (v == 0) {
        if (!allowzero) {
          if ((int64_t)i >= Rin) throw std::runtime_error("vkcnn: Reshape: zero-dim copies from out-of-range input axis");
          out.push_back(data_shape_dims[i]);
        } else {
          out.push_back(Dim::Const(0));
        }
      } else if (v > 0) {
        out.push_back(Dim::Const((uint64_t)v));
      } else {
        throw std::runtime_error("vkcnn: Reshape: negative dimension other than -1 is invalid");
      }
    }
    return out;
  };

  auto parse_shape_from_scalar_int = [&](const ScalarTensor &s) -> ShapeVector {
    int64_t v;
    switch (s.dtype) {
      case Dtype::Int64:  v = s.v.i; break;
      case Dtype::Int32: case Dtype::Int16: case Dtype::Int8: v = (int64_t)s.v.i; break;
      case Dtype::Uint64:
        if (s.v.u > (uint64_t)std::numeric_limits<int64_t>::max())
          throw std::runtime_error("vkcnn: Reshape: scalar shape too large");
        v = (int64_t)s.v.u; break;
      case Dtype::Uint32: case Dtype::Uint16: case Dtype::Uint8: v = (int64_t)s.v.u; break;
      default: throw std::runtime_error("vkcnn: Reshape: scalar shape must be integer");
    }
    ShapeVector out;
    if (v == -1) {
      if (used_infer_neg1) throw std::runtime_error("vkcnn: Reshape: at most one -1 is allowed");
      used_infer_neg1 = true; infer_index = 0; out.push_back(Dim::Const(1)); // placeholder
    } else if (v == 0) {
      if (!allowzero) {
        // copy input dim[0]
        if (data_shape_dims.size() < 1)
          throw std::runtime_error("vkcnn: Reshape: zero-dim copies from out-of-range input axis");
        out.push_back(data_shape_dims[0]);
      } else {
        out.push_back(Dim::Const(0));
      }
    } else if (v > 0) {
      out.push_back(Dim::Const((uint64_t)v));
    } else {
      throw std::runtime_error("vkcnn: Reshape: negative dimension other than -1 is invalid");
    }
    return out;
  };

  // Build target dims (shape spec)
  ShapeVector target_dims;
  if (shape.isShape())       target_dims = parse_shape_from_shape_tensor(shape.shapeTensor());
  else if (shape.isRaw()) target_dims = parse_shape_from_raw_1d_int(shape.raw());
  else if (shape.isScalar())   target_dims = parse_shape_from_scalar_int(shape.scalar());
  else                         throw std::runtime_error("vkcnn: Reshape: unsupported shape input kind");

  // Infer -1 if present (requires known Nin and all other target dims const)
  if (used_infer_neg1) {
    if (!Nin_known)
      throw std::runtime_error("vkcnn: Reshape: cannot infer -1 with unknown input element count");
    uint64_t prod_known = 1;
    for (size_t i = 0; i < target_dims.size(); ++i) {
      if ((int64_t)i == infer_index) continue;
      if (!target_dims[i].isConst())
        throw std::runtime_error("vkcnn: Reshape: cannot infer -1 with symbolic dims");
      prod_known *= target_dims[i].value();
    }
    if (prod_known == 0) {
      if (Nin != 0)
        throw std::runtime_error("vkcnn: Reshape: cannot infer -1 when product of other dims is zero and input has nonzero elements");
      throw std::runtime_error("vkcnn: Reshape: -1 with zero product is ambiguous (unsupported)");
    }
    if (Nin % prod_known != 0)
      throw std::runtime_error("vkcnn: Reshape: element count not divisible to infer -1");
    target_dims[(size_t)infer_index] = Dim::Const(Nin / prod_known);
  }

  // If ALL target dims are constant and Nin is known, enforce element-count equality.
  bool all_const = true;
  uint64_t Nout = 1;
  for (const Dim &d : target_dims) {
    if (!d.isConst()) { all_const = false; break; }
    Nout *= d.value();
  }
  if (all_const && Nin_known && Nout != Nin) {
    throw std::runtime_error("vkcnn: Reshape: output element count does not match input");
  }

  // ================== DATA KINDS ==================

  // 1) ShapeTensor AS DATA: treat as flat vector of Dim; reshape is a NO-OP on values.
  if (data.isShape()) {
    // We *ignore* target rank/grouping and just ensure element-count compatibility (above).
    // Return the same vector of Dims.
    return { data };
  }

  // 2) ScalarTensor -> RawTensor, serialize single element, just change shape
  if (data.isScalar()) {
    const auto &s = data.scalar();
    RawTensor out;
    out.type = s.dtype;
    out.shape = target_dims.empty() ? ShapeTensor::Scalar()
                                    : ShapeTensor::Tensor(target_dims);
    out.raw.resize(dtype_size(s.dtype));
    switch (s.dtype) {
      case Dtype::Int64:   std::memcpy(out.raw.data(), &s.v.i,       sizeof(std::int64_t)); break;
      case Dtype::Int32:   std::memcpy(out.raw.data(), &s.v.i,       sizeof(std::int32_t)); break;
      case Dtype::Int16:   std::memcpy(out.raw.data(), &s.v.i,       sizeof(std::int16_t)); break;
      case Dtype::Int8:    std::memcpy(out.raw.data(), &s.v.i,       sizeof(std::int8_t));  break;
      case Dtype::Uint64:  std::memcpy(out.raw.data(), &s.v.u,       sizeof(std::uint64_t)); break;
      case Dtype::Uint32:  std::memcpy(out.raw.data(), &s.v.u,       sizeof(std::uint32_t)); break;
      case Dtype::Uint16:  std::memcpy(out.raw.data(), &s.v.u,       sizeof(std::uint16_t)); break;
      case Dtype::Uint8:   std::memcpy(out.raw.data(), &s.v.u,       sizeof(std::uint8_t));  break;
      case Dtype::Float64: std::memcpy(out.raw.data(), &s.v.float64, sizeof(f64)); break;
      case Dtype::Float32: std::memcpy(out.raw.data(), &s.v.float32, sizeof(f32)); break;
      case Dtype::Float16: std::memcpy(out.raw.data(), &s.v.float16, sizeof(f16)); break;
      default: throw std::runtime_error("vkcnn: Reshape: unsupported scalar dtype");
    }
    return { Tensor::Raw(std::move(out)) };
  }

  // 3) RawTensor: change only shape metadata, keep bytes
  if (data.isRaw()) {
    const auto &rt = data.raw();
    RawTensor out;
    out.type  = rt.type;
    out.shape = target_dims.empty() ? ShapeTensor::Scalar()
                                    : ShapeTensor::Tensor(target_dims);
    out.raw   = rt.raw;
    return { Tensor::Raw(std::move(out)) };
  }

  // should not reach
  throw std::runtime_error("vkcnn: Reshape: unsupported data kind");
}

} // namespace vkcnn::details
