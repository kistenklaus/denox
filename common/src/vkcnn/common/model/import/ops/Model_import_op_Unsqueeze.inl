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
import_op_Unsqueeze([[maybe_unused]] ImportState &state,
                    std::span<const std::optional<Tensor>> inputs,
                    std::size_t outputCount,
                    const std::unordered_map<std::string, Tensor> &attributes,
                    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  // ---- contract
  if (outputCount != 1) {
    throw std::runtime_error(
        "vkcnn: Unsqueeze must produce exactly one output");
  }
  if (inputs.empty() || !inputs[0].has_value()) {
    throw std::runtime_error("vkcnn: Unsqueeze requires data input");
  }

  const Tensor &data = *inputs[0];

  // ---- reject unsupported kinds for data
  if (data.isRuntimeTensor()) {
    throw std::runtime_error(
        "vkcnn: Unsqueeze on runtime tensors is not supported");
  }
  if (data.isString()) {
    throw std::runtime_error(
        "vkcnn: Unsqueeze on string tensors is not supported");
  }
  if (data.isUnknown()) {
    throw std::runtime_error("vkcnn: Unsqueeze on unknown input kind");
  }

  // ---- parse axes (input[1] for opset >= 13; else attribute 'axes')
  auto parse_axes_tensor = [&](const Tensor &t) -> std::vector<int64_t> {
    // Scalars -> single axis
    if (t.isScalar()) {
      const auto &s = t.scalar();
      if (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
          s.dtype == Dtype::Int16) {
        return {static_cast<int64_t>(s.v.i)};
      }
      if (s.dtype == Dtype::Uint64 || s.dtype == Dtype::Uint32 ||
          s.dtype == Dtype::Uint16 || s.dtype == Dtype::Uint8) {
        return {static_cast<int64_t>(s.v.u)};
      }
      throw std::runtime_error("vkcnn: Unsqueeze axes scalar must be integer");
    }
    // List of integer scalars
    if (t.isList()) {
      std::vector<int64_t> out;
      out.reserve(t.list().tensors.size());
      for (const Tensor &e : t.list().tensors) {
        if (!e.isScalar())
          throw std::runtime_error(
              "vkcnn: Unsqueeze axes list elements must be integer scalars");
        const auto &s = e.scalar();
        if (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
            s.dtype == Dtype::Int16)
          out.push_back(static_cast<int64_t>(s.v.i));
        else if (s.dtype == Dtype::Uint64 || s.dtype == Dtype::Uint32 ||
                 s.dtype == Dtype::Uint16 || s.dtype == Dtype::Uint8)
          out.push_back(static_cast<int64_t>(s.v.u));
        else
          throw std::runtime_error(
              "vkcnn: Unsqueeze axes list elements must be integer");
      }
      return out;
    }
    // Constant tensor (0D or 1D) of integers
    if (t.isRaw()) {
      const auto &rt = t.raw();
      if (!(rt.type == Dtype::Int64 || rt.type == Dtype::Int32 ||
            rt.type == Dtype::Int16 || rt.type == Dtype::Uint64 ||
            rt.type == Dtype::Uint32 || rt.type == Dtype::Uint16 ||
            rt.type == Dtype::Uint8)) {
        throw std::runtime_error(
            "vkcnn: Unsqueeze axes tensor must be integer");
      }
      // compute element count
      size_t count = 1;
      if (rt.shape.isTensor()) {
        const auto &sv = rt.shape.dims();
        if (sv.size() > 1)
          throw std::runtime_error(
              "vkcnn: Unsqueeze axes input must be 0D or 1D");
        if (!sv.empty()) {
          if (!sv[0].isConst())
            throw std::runtime_error(
                "vkcnn: Unsqueeze axes length must be static");
          count = static_cast<size_t>(sv[0].value());
        }
      }
      size_t eb = (rt.type == Dtype::Int64 || rt.type == Dtype::Uint64)   ? 8
                  : (rt.type == Dtype::Int32 || rt.type == Dtype::Uint32) ? 4
                                                                          : 2;
      if (rt.raw.size() != count * eb)
        throw std::runtime_error("vkcnn: Unsqueeze axes raw size mismatch");
      std::vector<int64_t> out(count);
      const uint8_t *p = reinterpret_cast<const uint8_t *>(rt.raw.data());
      for (size_t i = 0; i < count; ++i) {
        switch (rt.type) {
        case Dtype::Int64:
          out[i] = reinterpret_cast<const int64_t *>(p)[i];
          break;
        case Dtype::Uint64:
          out[i] =
              static_cast<int64_t>(reinterpret_cast<const uint64_t *>(p)[i]);
          break;
        case Dtype::Int32:
          out[i] = reinterpret_cast<const int32_t *>(p)[i];
          break;
        case Dtype::Uint32:
          out[i] =
              static_cast<int64_t>(reinterpret_cast<const uint32_t *>(p)[i]);
          break;
        case Dtype::Int16:
          out[i] = reinterpret_cast<const int16_t *>(p)[i];
          break;
        case Dtype::Uint16:
          out[i] =
              static_cast<int64_t>(reinterpret_cast<const uint16_t *>(p)[i]);
          break;
        case Dtype::Uint8:
          out[i] =
              static_cast<int64_t>(reinterpret_cast<const uint8_t *>(p)[i]);
          break;
        default:
          break;
        }
      }
      return out;
    }
    // We do not support dynamic axes (Runtime/Shape/etc.)
    throw std::runtime_error("vkcnn: Unsqueeze axes must be constant integers");
  };

  auto get_axes = [&]() -> std::vector<int64_t> {
    // Prefer input[1] when provided (opset >= 13), else attribute "axes"
    if (inputs.size() >= 2 && inputs[1].has_value()) {
      return parse_axes_tensor(*inputs[1]);
    }
    auto it = attributes.find("axes");
    if (it != attributes.end()) {
      return parse_axes_tensor(it->second);
    }
    throw std::runtime_error(
        "vkcnn: Unsqueeze requires 'axes' (input or attribute)");
  };

  std::vector<int64_t> axes = get_axes();

  // ---- normalize & validate axes (relative to OUTPUT rank!)
  auto normalize_axes = [&](int64_t in_rank,
                            std::vector<int64_t> a) -> std::vector<int64_t> {
    const int64_t out_rank = in_rank + static_cast<int64_t>(a.size());
    // normalize negatives
    for (auto &v : a) {
      if (v < 0)
        v += out_rank;
      if (v < 0 || v >= out_rank) {
        throw std::runtime_error(
            "vkcnn: Unsqueeze 'axes' value out of range for output rank");
      }
    }
    // check duplicates
    std::vector<char> seen(static_cast<size_t>(out_rank), 0);
    for (auto v : a) {
      if (seen[static_cast<size_t>(v)]) {
        throw std::runtime_error("vkcnn: Unsqueeze 'axes' contains duplicates");
      }
      seen[static_cast<size_t>(v)] = 1;
    }
    return a;
  };

  // ---- helper to build the new dim vector by inserting 1's at axes
  auto build_unsqueezed_dims =
      [&](const ShapeVector &in_dims,
          const std::vector<int64_t> &axes_norm) -> ShapeVector {
    const int64_t in_rank = static_cast<int64_t>(in_dims.size());
    const int64_t out_rank = in_rank + static_cast<int64_t>(axes_norm.size());
    std::vector<char> mark(static_cast<size_t>(out_rank), 0);
    for (auto a : axes_norm)
      mark[static_cast<size_t>(a)] = 1;

    ShapeVector out;
    out.reserve(static_cast<size_t>(out_rank));
    int64_t src = 0;
    for (int64_t i = 0; i < out_rank; ++i) {
      if (mark[static_cast<size_t>(i)]) {
        out.push_back(Dim::Const(1));
      } else {
        out.push_back(in_dims[static_cast<size_t>(src++)]);
      }
    }
    assert(src == in_rank);
    return out;
  };

  // ========================= CASE 1: ShapeTensor ===========================
  if (data.isShape()) {
    const auto &sv = data.shapeTensor();
    // Determine "input rank" as the length of the shape-vector (scalar => rank
    // 0)
    int64_t in_rank = sv.isScalar() ? 0 : static_cast<int64_t>(sv.dims().size());
    std::vector<int64_t> axes_norm = normalize_axes(in_rank, axes);

    // For scalar ShapeTensor: its "dims list" is empty
    const ShapeVector &in_dims = sv.isScalar()
                                     ? *reinterpret_cast<const ShapeVector *>(
                                           &sv) // never used; we’ll branch
                                     : sv.dims();

    ShapeVector out_dims = sv.isScalar()
                               ? build_unsqueezed_dims(ShapeVector{}, axes_norm)
                               : build_unsqueezed_dims(in_dims, axes_norm);

    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out_dims)))};
  }

  // ========================= CASE 2: RawTensor (constants) =================
  if (data.isRaw()) {
    const auto &rt = data.raw();

    // We don’t support string raw for reshapes here
    if (rt.type == Dtype::String) {
      throw std::runtime_error(
          "vkcnn: Unsqueeze on RawTensor<string> is not supported");
    }

    // Compute input rank and dims
    ShapeVector in_dims_vec;
    int64_t in_rank = 0;
    if (rt.shape.isScalar()) {
      in_dims_vec = ShapeVector{}; // rank 0
      in_rank = 0;
    } else {
      in_dims_vec = rt.shape.dims();
      in_rank = static_cast<int64_t>(in_dims_vec.size());
      // ensure static shape
      for (const Dim &d : in_dims_vec) {
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: Unsqueeze RawTensor requires static shape");
      }
    }

    std::vector<int64_t> axes_norm = normalize_axes(in_rank, axes);
    ShapeVector out_dims = build_unsqueezed_dims(in_dims_vec, axes_norm);

    RawTensor out;
    out.type = rt.type;
    out.shape = out_dims.empty() ? ShapeTensor::Scalar()
                                 : ShapeTensor::Tensor(std::move(out_dims));
    out.raw = rt.raw; // reshape only, data unchanged

    return {Tensor::Raw(std::move(out))};
  }

  // ========================= CASE 3: ScalarTensor ==========================
  if (data.isScalar()) {
    const auto &s = data.scalar();

    // Only numeric scalar supported here (no strings)
    bool is_num = (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
                   s.dtype == Dtype::Int16 || s.dtype == Dtype::Int8 ||
                   s.dtype == Dtype::Uint64 || s.dtype == Dtype::Uint32 ||
                   s.dtype == Dtype::Uint16 || s.dtype == Dtype::Uint8 ||
                   s.dtype == Dtype::Float32 || s.dtype == Dtype::Float64 ||
                   s.dtype == Dtype::Float16);

    if (!is_num) {
      throw std::runtime_error(
          "vkcnn: Unsqueeze on non-numeric scalar is not supported");
    }

    // Scalar has rank 0 → normalize axes accordingly and build shape of ones
    std::vector<int64_t> axes_norm = normalize_axes(/*in_rank=*/0, axes);

    ShapeVector out_dims = build_unsqueezed_dims(ShapeVector{}, axes_norm);

    // Serialize the single scalar into RawTensor
    RawTensor out;
    out.type = s.dtype;
    out.shape = out_dims.empty() ? ShapeTensor::Scalar()
                                 : ShapeTensor::Tensor(std::move(out_dims));
    out.raw.resize(s.dtype == Dtype::Float64 || s.dtype == Dtype::Uint64 ||
                           s.dtype == Dtype::Int64
                       ? 8
                   : s.dtype == Dtype::Float32 || s.dtype == Dtype::Uint32 ||
                           s.dtype == Dtype::Int32
                       ? 4
                   : s.dtype == Dtype::Float16 || s.dtype == Dtype::Uint16 ||
                           s.dtype == Dtype::Int16
                       ? 2
                       : 1);

    // copy the scalar bytes
    switch (s.dtype) {
    case Dtype::Int64:
      std::memcpy(out.raw.data(), &s.v.i, sizeof(std::int64_t));
      break;
    case Dtype::Int32:
      std::memcpy(out.raw.data(), &s.v.i, sizeof(std::int32_t));
      break;
    case Dtype::Int16:
      std::memcpy(out.raw.data(), &s.v.i, sizeof(std::int16_t));
      break;
    case Dtype::Int8:
      std::memcpy(out.raw.data(), &s.v.i, sizeof(std::int8_t));
      break;
    case Dtype::Uint64:
      std::memcpy(out.raw.data(), &s.v.u, sizeof(std::uint64_t));
      break;
    case Dtype::Uint32:
      std::memcpy(out.raw.data(), &s.v.u, sizeof(std::uint32_t));
      break;
    case Dtype::Uint16:
      std::memcpy(out.raw.data(), &s.v.u, sizeof(std::uint16_t));
      break;
    case Dtype::Uint8:
      std::memcpy(out.raw.data(), &s.v.u, sizeof(std::uint8_t));
      break;
    case Dtype::Float64:
      std::memcpy(out.raw.data(), &s.v.float64, sizeof(f64));
      break;
    case Dtype::Float32:
      std::memcpy(out.raw.data(), &s.v.float32, sizeof(f32));
      break;
    case Dtype::Float16:
      std::memcpy(out.raw.data(), &s.v.float16, sizeof(f16));
      break;
    default:
      break;
    }

    return {Tensor::Raw(std::move(out))};
  }

  // Anything else -> unsupported for now
  throw std::runtime_error(fmt::format(
      "vkcnn: Unsqueeze not supported for given input kind (node='{}')",
      node.name()));
}

} // namespace vkcnn::details
