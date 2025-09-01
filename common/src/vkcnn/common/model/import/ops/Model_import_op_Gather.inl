#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Gather(
    [[maybe_unused]] ImportState &state,
    std::span<const std::optional<Tensor>> inputs, std::size_t outputCount,
    const std::unordered_map<std::string, Tensor> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // --- Contract checks
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Gather requires exactly two inputs (data, "
                    "indices). Got {} (node='{}')",
                    inputs.size(), node.op_type()));
  }
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Gather must produce exactly one output");
  }

  const Tensor &data = *inputs[0];
  const Tensor &indices = *inputs[1];

  // --- Reject runtime tensors anywhere
  if (data.isRuntimeTensor()) {
    throw std::runtime_error(
        "vkcnn: Gather on runtime tensors is not supported");
  }
  if (indices.isRuntimeTensor()) {
    throw std::runtime_error(
        "vkcnn: Gather with runtime indices is not supported");
  }

  // --- Parse axis (default 0). Your policy: present-but-Unknown => 0.
  auto get_axis = [&]() -> int64_t {
    auto it = attributes.find("axis");
    if (it == attributes.end())
      return 0;
    const Tensor &a = it->second;
    if (a.isUnknown())
      return 0;
    if (!a.isScalar())
      throw std::runtime_error(
          "vkcnn: Gather attribute 'axis' must be an integer scalar");
    const auto &s = a.scalar();
    if (s.dtype != Dtype::Int64 && s.dtype != Dtype::Int32 &&
        s.dtype != Dtype::Int16)
      throw std::runtime_error(
          "vkcnn: Gather attribute 'axis' must be an integer");
    return s.v.i;
  };
  int64_t axis = get_axis();

  // --- Helpers -------------------------------------------------------------

  // Read indices as a flat vector<int64_t> and remember their shape (for
  // RawTensor path).
  struct IndicesInfo {
    bool is_scalar = false;      // true if scalar index
    ShapeTensor shape;           // shape of indices tensor (scalar => Scalar())
    std::vector<int64_t> values; // flattened values
  };

  auto parse_indices_any = [&](const Tensor &t,
                               bool allow_multi_rank) -> IndicesInfo {
    // Scalar ints
    if (t.isScalar()) {
      const auto &s = t.scalar();
      if (s.dtype != Dtype::Int64 && s.dtype != Dtype::Int32 &&
          s.dtype != Dtype::Int16)
        throw std::runtime_error(
            "vkcnn: Gather indices scalar must be integer");
      return IndicesInfo{true, ShapeTensor::Scalar(),
                         std::vector<int64_t>{static_cast<int64_t>(s.v.i)}};
    }
    // List of integer scalars -> 1-D vector
    if (t.isList()) {
      const auto &lst = t.list().tensors;
      IndicesInfo out;
      out.is_scalar = false;
      out.values.reserve(lst.size());
      for (const Tensor &e : lst) {
        if (!e.isScalar())
          throw std::runtime_error(
              "vkcnn: Gather indices list elements must be integer scalars");
        const auto &s = e.scalar();
        if (s.dtype != Dtype::Int64 && s.dtype != Dtype::Int32 &&
            s.dtype != Dtype::Int16)
          throw std::runtime_error(
              "vkcnn: Gather indices list elements must be integer");
        out.values.push_back(static_cast<int64_t>(s.v.i));
      }
      out.shape = ShapeTensor::Vec(out.values.size()); // 1-D [n]
      return out;
    }
    // Constant tensor of ints
    if (t.isRaw()) {
      const auto &rt = t.raw();
      // dtype check (support int64/int32/int16)
      if (!(rt.type == Dtype::Int64 || rt.type == Dtype::Int32 ||
            rt.type == Dtype::Int16)) {
        throw std::runtime_error(
            "vkcnn: Gather indices tensor must be int64/int32/int16");
      }
      // Compute element count from shape
      size_t count = 1;
      if (rt.shape.isTensor()) {
        for (const Dim &d : rt.shape.dims()) {
          if (!d.isConst())
            throw std::runtime_error(
                "vkcnn: Gather indices shape must be static");
          count *= static_cast<size_t>(d.value());
        }
      } else {
        // scalar
        count = 1;
      }
      // Read raw -> values
      size_t elem_bytes =
          (rt.type == Dtype::Int64) ? 8 : (rt.type == Dtype::Int32 ? 4 : 2);
      if (rt.raw.size() != count * elem_bytes)
        throw std::runtime_error(
            "vkcnn: Gather indices raw payload size mismatch");
      IndicesInfo out;
      out.values.resize(count);
      if (rt.type == Dtype::Int64) {
        const int64_t *p = reinterpret_cast<const int64_t *>(rt.raw.data());
        for (size_t i = 0; i < count; ++i)
          out.values[i] = p[i];
      } else if (rt.type == Dtype::Int32) {
        const int32_t *p = reinterpret_cast<const int32_t *>(rt.raw.data());
        for (size_t i = 0; i < count; ++i)
          out.values[i] = static_cast<int64_t>(p[i]);
      } else { // Int16
        const int16_t *p = reinterpret_cast<const int16_t *>(rt.raw.data());
        for (size_t i = 0; i < count; ++i)
          out.values[i] = static_cast<int64_t>(p[i]);
      }
      out.is_scalar = (rt.shape.isScalar());
      out.shape = rt.shape.isScalar()
                      ? ShapeTensor::Scalar()
                      : ShapeTensor::Tensor(rt.shape.dims()); // preserve rank
      if (!allow_multi_rank && out.shape.isTensor() &&
          out.shape.dims().size() > 1) {
        throw std::runtime_error(
            "vkcnn: Gather indices must be scalar or 1-D for this input kind");
      }
      return out;
    }

    throw std::runtime_error("vkcnn: Gather indices must be integer scalars, "
                             "integer lists, or constant integer tensors");
  };

  auto normalize_axis = [](int64_t ax, int64_t rank) -> int64_t {
    if (rank < 0)
      throw std::logic_error("vkcnn: internal error rank<0");
    if (ax < 0)
      ax += rank;
    if (ax < 0 || ax >= rank)
      throw std::runtime_error("vkcnn: Gather 'axis' out of range");
    return ax;
  };

  auto normalize_idx = [](int64_t idx, int64_t dim) -> int64_t {
    if (idx < 0)
      idx += dim;
    if (idx < 0 || idx >= dim)
      throw std::runtime_error("vkcnn: Gather index out of bounds");
    return idx;
  };

  // ===================== CASE A: ShapeTensor ===============================
  if (data.isShape()) {
    const ShapeTensor &sv = data.shapeTensor();
    // ShapeTensor in your IR is always 1-D vector of Dims (for "shape values").
    // Axis must be 0 for a 1-D vector.
    int64_t rank = 1;
    int64_t ax = normalize_axis(axis, rank);
    (void)ax; // ax==0 guaranteed

    // Indices: for ShapeTensor we only support scalar or 1-D (so we can return
    // another ShapeTensor)
    IndicesInfo idx = parse_indices_any(indices, /*allow_multi_rank=*/false);

    const auto &in =
        sv.isScalar()
            ? ShapeVector{}
            // theoretically empty; but Gather on scalar is invalid anyway
            : sv.dims();

    if (sv.isScalar()) {
      throw std::runtime_error(
          "vkcnn: Gather on a scalar ShapeTensor is invalid");
    }

    if (idx.is_scalar) {
      int64_t n = static_cast<int64_t>(in.size());
      int64_t ii = normalize_idx(idx.values[0], n);
      // NOTE: ONNX would return a scalar int64 here.
      // Your ShapeTensor cannot carry a single Dim in Scalar() form,
      // so we return a 1-D shape-vector of length 1 to preserve the Dim.
      ShapeVector out(1);
      out[0] = in[static_cast<size_t>(ii)];
      return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
    } else {
      // 1-D indices -> 1-D shape-vector of selected dims
      if (!idx.shape.isTensor())
        throw std::logic_error("vkcnn: indices parsing bug (expected 1-D)");
      ShapeVector out;
      out.reserve(idx.values.size());
      int64_t n = static_cast<int64_t>(in.size());
      for (int64_t v : idx.values) {
        int64_t ii = normalize_idx(v, n);
        out.push_back(in[static_cast<size_t>(ii)]);
      }
      return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
    }
  }

  // ===================== CASE B: ListTensor (sequence) =====================
  if (data.isList()) {
    // Sequence is conceptually 1-D; only axis==0 makes sense here.
    int64_t ax = normalize_axis(axis, /*rank=*/1);
    (void)ax; // must be 0

    const auto &in = data.list().tensors;
    IndicesInfo idx = parse_indices_any(indices, /*allow_multi_rank=*/false);

    if (idx.is_scalar) {
      int64_t n = static_cast<int64_t>(in.size());
      int64_t ii = normalize_idx(idx.values[0], n);
      // Return the selected element directly (sequence -> element)
      return {in[static_cast<size_t>(ii)]};
    } else {
      // 1-D indices -> subsequence
      if (!idx.shape.isTensor())
        throw std::logic_error("vkcnn: indices parsing bug (expected 1-D)");
      std::vector<Tensor> out;
      out.reserve(idx.values.size());
      int64_t n = static_cast<int64_t>(in.size());
      for (int64_t v : idx.values) {
        int64_t ii = normalize_idx(v, n);
        out.push_back(in[static_cast<size_t>(ii)]);
      }
      return {Tensor::List(std::move(out))};
    }
  }

  // ===================== CASE C: RawTensor (constant tensor) ===============
  if (data.isRaw()) {
    const auto &rt = data.raw();

    // Strings in RawTensor are not byte-addressable in a standard way here.
    if (rt.type == Dtype::String) {
      throw std::runtime_error(
          "vkcnn: Gather on RawTensor<string> not supported");
    }

    // Compute rank and sizes (must be static)
    std::vector<int64_t> sizes;
    int64_t rank;
    if (rt.shape.isScalar()) {
      rank = 0;
    } else {
      rank = static_cast<int64_t>(rt.shape.dims().size());
      sizes.reserve(static_cast<size_t>(rank));
      for (const Dim &d : rt.shape.dims()) {
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: Gather on RawTensor requires static shape");
        sizes.push_back(static_cast<int64_t>(d.value()));
      }
    }
    if (rank == 0) {
      throw std::runtime_error("vkcnn: Gather on scalar tensor is invalid");
    }

    int64_t ax = normalize_axis(axis, rank);

    // Parse indices (allow full-rank for true tensor gather)
    IndicesInfo idx = parse_indices_any(indices, /*allow_multi_rank=*/true);

    // Compute elem size (bytes)
    auto dtype_size = [](Dtype t) -> size_t {
      switch (t) {
      case Dtype::Int8:
      case Dtype::Uint8:
        return 1;
      case Dtype::Int16:
      case Dtype::Uint16:
      case Dtype::Float16:
        return 2;
      case Dtype::Int32:
      case Dtype::Uint32:
      case Dtype::Float32:
        return 4;
      case Dtype::Int64:
      case Dtype::Uint64:
      case Dtype::Float64:
        return 8;
      default:
        return 0;
      }
    };
    size_t esize = dtype_size(rt.type);
    if (esize == 0)
      throw std::runtime_error(
          "vkcnn: Gather on RawTensor: unsupported dtype for raw gather");

    // Precompute products
    auto prod = [](const std::vector<int64_t> &v, size_t a,
                   size_t b) -> int64_t {
      int64_t p = 1;
      for (size_t i = a; i < b; ++i)
        p *= v[i];
      return p;
    };

    const int64_t inner_block_elems =
        (ax + 1 <= rank) ? prod(sizes, static_cast<size_t>(ax + 1),
                                static_cast<size_t>(rank))
                         : 1;
    const int64_t middle = sizes[static_cast<size_t>(ax)];
    const int64_t outer_block =
        (ax > 0) ? prod(sizes, 0, static_cast<size_t>(ax)) : 1;

    // Normalize & validate indices against 'middle'
    std::vector<int64_t> idx_vals = idx.values;
    for (auto &v : idx_vals)
      v = normalize_idx(v, middle);

    // Output shape: S[:ax] + K + S[ax+1:]
    ShapeVector out_shape_vec;
    // prefix
    for (int64_t i = 0; i < ax; ++i)
      out_shape_vec.push_back(
          Dim::Const(static_cast<uint64_t>(sizes[static_cast<size_t>(i)])));
    // K
    if (!idx.is_scalar) {
      if (idx.shape.isScalar()) {
        // logically scalar -> nothing; but we only expect scalar/nd here
      } else {
        for (const Dim &kd : idx.shape.dims())
          out_shape_vec.push_back(kd); // all K dims are constants here
      }
    }
    // suffix
    for (int64_t i = ax + 1; i < rank; ++i)
      out_shape_vec.push_back(
          Dim::Const(static_cast<uint64_t>(sizes[static_cast<size_t>(i)])));

    // Compute element counts
    int64_t out_elems_per_outer =
        (idx.is_scalar ? 1 : static_cast<int64_t>(idx_vals.size())) *
        inner_block_elems;
    int64_t out_outer = outer_block;
    size_t out_total_elems =
        static_cast<size_t>(out_elems_per_outer * out_outer);

    // Reserve output raw buffer
    RawTensor out;
    out.type = rt.type;
    out.shape = out_shape_vec.empty()
                    ? ShapeTensor::Scalar()
                    : ShapeTensor::Tensor(std::move(out_shape_vec));
    out.raw.resize(static_cast<size_t>(out_total_elems) * esize);

    const uint8_t *src = reinterpret_cast<const uint8_t *>(rt.raw.data());
    uint8_t *dst = reinterpret_cast<uint8_t *>(out.raw.data());

    const size_t copy_bytes = static_cast<size_t>(inner_block_elems) * esize;
    const size_t stride_bytes =
        static_cast<size_t>(middle) *
        copy_bytes; // step between successive indices for same outer

    // Copy: for each outer-block, gather selected slices along ax
    size_t dst_off = 0;
    for (int64_t ob = 0; ob < outer_block; ++ob) {
      const size_t base = static_cast<size_t>(ob) * stride_bytes;
      if (idx.is_scalar) {
        const size_t src_off =
            base + static_cast<size_t>(idx_vals[0]) * copy_bytes;
        std::memcpy(dst + dst_off, src + src_off, copy_bytes);
        dst_off += copy_bytes;
      } else {
        for (int64_t v : idx_vals) {
          const size_t src_off = base + static_cast<size_t>(v) * copy_bytes;
          std::memcpy(dst + dst_off, src + src_off, copy_bytes);
          dst_off += copy_bytes;
        }
      }
    }

    return {Tensor::Raw(std::move(out))};
  }

  // ===================== CASE D: Scalar / String ===========================
  if (data.isScalar() || data.isString()) {
    // Scalar/string has rank 0 -> Gather is invalid for any axis.
    throw std::runtime_error("vkcnn: Gather on scalar/string input is invalid");
  }

  if (data.isUnknown()) {
    throw std::runtime_error("vkcnn: Gather on unknown input kind");
  }

  // Fallback (shouldn't get here)
  throw std::runtime_error(
      fmt::format("vkcnn: operation Gather is not supported for this input "
                  "kind (node='{}')",
                  node.name()));
}
} // namespace vkcnn::details
