#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Slice(
    [[maybe_unused]] ImportState &state,
    std::span<const std::optional<Tensor>> inputs, std::size_t outputCount,
    const std::unordered_map<std::string, Tensor> &attributes,
    opset_version /*version*/, [[maybe_unused]] const onnx::NodeProto &node) {

  // ---------- contract ----------
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Slice must produce exactly one output");
  }
  if (inputs.size() < 3) {
    throw std::runtime_error(
        "vkcnn: Slice requires at least 3 inputs: data, starts, ends");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value()) {
      throw std::runtime_error("vkcnn: Slice input present but not provided");
    }
  }

  const Tensor &data = *inputs[0];
  const Tensor &t_st = *inputs[1];
  const Tensor &t_en = *inputs[2];
  const Tensor *t_ax =
      (inputs.size() >= 4 && inputs[3].has_value()) ? &*inputs[3] : nullptr;
  const Tensor *t_stp =
      (inputs.size() >= 5 && inputs[4].has_value()) ? &*inputs[4] : nullptr;

  // Fallback to attributes for older models (starts/ends/axes/steps)
  auto attr_or = [&](const char *key) -> const Tensor * {
    auto it = attributes.find(key);
    return it == attributes.end() ? nullptr : &it->second;
  };
  if (!t_ax)
    t_ax = attr_or("axes");
  if (!t_stp)
    t_stp = attr_or("steps");

  // ---------- reject unsupported kinds ----------
  auto bad_kind = [&](const Tensor &t) -> bool {
    return t.isRuntimeTensor() || t.isString() || t.isList() || t.isUnknown();
  };
  if (bad_kind(data))
    throw std::runtime_error("vkcnn: Slice: unsupported data kind");
  if (bad_kind(t_st))
    throw std::runtime_error("vkcnn: Slice: unsupported 'starts' kind");
  if (bad_kind(t_en))
    throw std::runtime_error("vkcnn: Slice: unsupported 'ends' kind");
  if (t_ax && bad_kind(*t_ax))
    throw std::runtime_error("vkcnn: Slice: unsupported 'axes' kind");
  if (t_stp && bad_kind(*t_stp))
    throw std::runtime_error("vkcnn: Slice: unsupported 'steps' kind");

  // ---------- helpers to read integer vectors ----------
  auto read_int_list = [&](const Tensor &t,
                           const char *what) -> std::vector<int64_t> {
    std::vector<int64_t> out;
    if (t.isScalar()) {
      const auto &s = t.scalar();
      switch (s.dtype) {
      case Dtype::Int64:
        out.push_back(s.v.i);
        break;
      case Dtype::Int32:
      case Dtype::Int16:
      case Dtype::Int8:
        out.push_back((int64_t)s.v.i);
        break;
      case Dtype::Uint64:
        out.push_back((int64_t)s.v.u);
        break;
      case Dtype::Uint32:
      case Dtype::Uint16:
      case Dtype::Uint8:
        out.push_back((int64_t)s.v.u);
        break;
      default:
        throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                                 " must be integer");
      }
      return out;
    }
    if (t.isRaw()) {
      const auto &rt = t.raw();
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
      if (!is_int_dt(rt.type))
        throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                                 " tensor must be integer dtype");
      size_t n = 1;
      if (rt.shape.isTensor()) {
        const auto &sv = rt.shape.dims();
        if (sv.size() != 1)
          throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                                   " must be 1-D or scalar");
        if (!sv[0].isConst())
          throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                                   " length must be static");
        n = (size_t)sv[0].value();
      }
      const size_t esz = dtype_size(rt.type);
      if (rt.raw.size() != n * esz)
        throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                                 " raw payload size mismatch");
      out.resize(n);
      const uint8_t *p = (const uint8_t *)rt.raw.data();
      for (size_t i = 0; i < n; ++i) {
        switch (rt.type) {
        case Dtype::Int64:
          out[i] = ((const int64_t *)p)[i];
          break;
        case Dtype::Int32:
          out[i] = (int64_t)((const int32_t *)p)[i];
          break;
        case Dtype::Int16:
          out[i] = (int64_t)((const int16_t *)p)[i];
          break;
        case Dtype::Int8:
          out[i] = (int64_t)((const int8_t *)p)[i];
          break;
        case Dtype::Uint64:
          out[i] = (int64_t)((const uint64_t *)p)[i];
          break;
        case Dtype::Uint32:
          out[i] = (int64_t)((const uint32_t *)p)[i];
          break;
        case Dtype::Uint16:
          out[i] = (int64_t)((const uint16_t *)p)[i];
          break;
        case Dtype::Uint8:
          out[i] = (int64_t)((const uint8_t *)p)[i];
          break;
        default:
          break;
        }
      }
      return out;
    }
    throw std::runtime_error(std::string("vkcnn: Slice: ") + what +
                             " must be scalar or 1-D integer tensor");
  };

  // ---------- read data "rank" and sizes ----------
  int64_t rank = 0;
  std::vector<int64_t>
      in_sizes; // for Raw: static sizes; for ShapeTensor-as-data: [L]
  if (data.isScalar()) {
    rank = 0;
  } else if (data.isRaw()) {
    const auto &rt = data.raw();
    if (rt.shape.isScalar()) {
      rank = 0;
    } else {
      rank = (int64_t)rt.shape.dims().size();
      in_sizes.resize((size_t)rank);
      for (size_t i = 0; i < (size_t)rank; ++i) {
        const Dim &d = rt.shape.dims()[i];
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: Slice on RawTensor requires static shapes");
        in_sizes[i] = (int64_t)d.value();
      }
    }
  } else if (data.isShape()) {
    // ShapeTensor as DATA: treat as 1-D vector of length L (even if L==0)
    rank = 1;
    if (data.shapeTensor().isScalar()) {
      in_sizes = {0};
    } else {
      in_sizes = {(int64_t)data.shapeTensor().dims().size()};
    }
  } else {
    throw std::runtime_error("vkcnn: Slice: unsupported data kind");
  }

  // ---------- read starts/ends/axes/steps ----------
  std::vector<int64_t> starts = read_int_list(t_st, "starts");
  std::vector<int64_t> ends = read_int_list(t_en, "ends");

  std::vector<int64_t> axes;
  if (t_ax) {
    axes = read_int_list(*t_ax, "axes");
  } else {
    // ONNX default: axes = [0..rank-1]; but then starts/ends length must ==
    // rank
    if (rank < 0)
      throw std::runtime_error("vkcnn: Slice internal rank error");
    axes.resize((size_t)rank);
    for (int64_t i = 0; i < rank; ++i)
      axes[(size_t)i] = i;
    if (starts.size() != (size_t)rank || ends.size() != (size_t)rank) {
      // For ShapeTensor-as-data (rank==1), it's common to pass single
      // start/end; allow that.
      if (!(data.isShape() && rank == 1 && starts.size() == 1 &&
            ends.size() == 1)) {
        throw std::runtime_error("vkcnn: Slice: when 'axes' is omitted, "
                                 "starts/ends must match rank");
      }
      axes = {0};
    }
  }

  std::vector<int64_t> steps;
  if (t_stp) {
    steps = read_int_list(*t_stp, "steps");
  } else {
    steps.assign(starts.size(), 1);
  }

  // lengths must match
  const size_t m = starts.size();
  if (ends.size() != m)
    throw std::runtime_error(
        "vkcnn: Slice: 'starts' and 'ends' must match in length");
  if (axes.size() != m)
    throw std::runtime_error(
        "vkcnn: Slice: 'axes' length must match 'starts'/'ends'");
  if (steps.size() != m)
    throw std::runtime_error(
        "vkcnn: Slice: 'steps' length must match 'starts'/'ends'");

  // Scalar data: only support empty slice spec (no axes)
  if (data.isScalar()) {
    if (m != 0) {
      throw std::runtime_error(
          "vkcnn: Slice: cannot slice a scalar (axes must be empty)");
    }
    return {data}; // no-op
  }

  // Normalize axes: map negatives, check range & duplicates
  auto norm_axis = [&](int64_t a) -> int64_t { return a < 0 ? a + rank : a; };
  std::vector<char> used((size_t)std::max<int64_t>(rank, 1), 0);
  for (size_t i = 0; i < m; ++i) {
    int64_t a = norm_axis(axes[i]);
    if (a < 0 || a >= rank)
      throw std::runtime_error("vkcnn: Slice: axis out of range");
    if (used[(size_t)a])
      throw std::runtime_error("vkcnn: Slice: duplicate axes not allowed");
    used[(size_t)a] = 1;
    axes[i] = a;
  }

  // Steps must be non-zero (negative allowed => reverse slice)
  for (auto s : steps) {
    if (s == 0)
      throw std::runtime_error("vkcnn: Slice: steps must be non-zero");
  }

  // ---------- compute normalized start/end and output sizes (supports negative
  // steps) ----------
  std::vector<int64_t> starts_n(m), ends_n(m), steps_n = steps;
  std::vector<int64_t> out_sizes = in_sizes; // init with input sizes

  auto clamp = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
    return v < lo ? lo : (v > hi ? hi : v);
  };

  for (size_t k = 0; k < m; ++k) {
    const int64_t ax = axes[k];
    const int64_t dim = in_sizes[(size_t)ax];

    int64_t s = starts[k];
    int64_t e = ends[k];
    int64_t st = steps_n[k];

    // translate negatives relative to dim (numpy-style)
    if (s < 0)
      s += dim;
    if (e < 0)
      e += dim;

    if (st > 0) {
      // forward slice
      s = clamp(s, 0, dim);
      e = clamp(e, 0, dim);
      int64_t diff = e - s;
      int64_t len = diff <= 0 ? 0 : ((diff + st - 1) / st);
      out_sizes[(size_t)ax] = len;
    } else { // st < 0 (reverse)
      const int64_t absst = -st;
      s = clamp(s, 0, dim - 1);
      e = clamp(e, -1, dim - 1);
      int64_t diff = s - e;
      int64_t len = diff <= 0 ? 0 : ((diff + absst - 1) / absst);
      out_sizes[(size_t)ax] = len;
    }

    starts_n[k] = s;
    ends_n[k] = e;
    steps_n[k] = st; // keep sign
  }

  // ---------- produce outputs for each data kind ----------
  // 1) ShapeTensor as DATA: treat as 1-D vector of Dim; slice along axis 0
  if (data.isShape()) {
    if (rank != 1)
      throw std::logic_error("vkcnn: internal (shape-as-data rank)");
    if (m != 1 || axes[0] != 0) {
      throw std::runtime_error(
          "vkcnn: Slice on ShapeTensor-as-data expects axes=[0]");
    }

    const int64_t L = in_sizes[0];
    ShapeVector out_vec;
    out_vec.reserve((size_t)out_sizes[0]);

    // empty vector => quick return
    if (L == 0) {
      return {Tensor::Shape(ShapeTensor::Tensor(std::move(out_vec)))};
    }
    const auto &vec_in = data.shapeTensor().dims();

    const int64_t s0 = starts_n[0], e0 = ends_n[0], st0 = steps_n[0];
    if (st0 > 0) {
      for (int64_t idx = s0; idx < e0; idx += st0) {
        out_vec.push_back(vec_in[(size_t)idx]);
      }
    } else { // reverse
      for (int64_t idx = s0; idx > e0; idx += st0) {
        out_vec.push_back(vec_in[(size_t)idx]);
      }
    }
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out_vec)))};
  }

  // 2) RawTensor: general N-D slicing with positive or negative steps
  if (data.isRaw()) {
    const auto &rt = data.raw();
    const size_t esz = dtype_size(rt.type);
    if (esz == 0)
      throw std::runtime_error("vkcnn: Slice unsupported dtype");

    // Compute output element count and build output ShapeVector
    int64_t Nout_el = 1;
    ShapeVector out_dims;
    if (rank == 0) {
      Nout_el = 1;
    } else {
      out_dims.reserve((size_t)rank);
      for (int64_t d = 0; d < rank; ++d) {
        int64_t v = out_sizes[(size_t)d];
        if (v < 0)
          v = 0;
        Nout_el *= v;
        out_dims.push_back(Dim::Const((uint64_t)v));
      }
    }

    RawTensor out;
    out.type = rt.type;
    out.shape =
        (rank == 0) ? ShapeTensor::Scalar() : ShapeTensor::Tensor(out_dims);
    out.raw.resize((size_t)std::max<int64_t>(0, Nout_el) * esz);
    if (Nout_el == 0) {
      return {Tensor::Raw(std::move(out))};
    }

    // Precompute strides for input tensor (row-major)
    std::vector<int64_t> in_strides((size_t)std::max<int64_t>(rank, 1), 1);
    for (int64_t d = rank - 2; d >= 0; --d) {
      in_strides[(size_t)d] =
          in_strides[(size_t)(d + 1)] * in_sizes[(size_t)(d + 1)];
    }

    // Build per-dimension slice parameters: start and step; for non-sliced axes
    // start=0, step=1
    std::vector<int64_t> per_axis_start((size_t)rank, 0);
    std::vector<int64_t> per_axis_step((size_t)rank, 1);
    for (size_t k = 0; k < m; ++k) {
      int64_t ax = axes[k];
      per_axis_start[(size_t)ax] = starts_n[k];
      per_axis_step[(size_t)ax] = steps_n[k]; // may be negative
    }

    // Convert linear out index -> multi-index -> input index
    auto lin_to_multi = [&](int64_t lin) -> std::vector<int64_t> {
      std::vector<int64_t> idx((size_t)rank, 0);
      for (int64_t d = rank - 1; d >= 0; --d) {
        int64_t size_d = out_sizes[(size_t)d];
        idx[(size_t)d] = size_d == 0 ? 0 : (lin % size_d);
        if (size_d != 0)
          lin /= size_d;
      }
      return idx;
    };

    const uint8_t *src = reinterpret_cast<const uint8_t *>(rt.raw.data());
    uint8_t *dst = reinterpret_cast<uint8_t *>(out.raw.data());

    for (int64_t lin = 0; lin < Nout_el; ++lin) {
      std::vector<int64_t> oidx = lin_to_multi(lin);
      // Map to input index
      int64_t in_off_el = 0;
      for (int64_t d = 0; d < rank; ++d) {
        int64_t in_i = per_axis_start[(size_t)d] +
                       oidx[(size_t)d] * per_axis_step[(size_t)d];
        in_off_el += in_i * in_strides[(size_t)d];
      }
      size_t in_off_b = (size_t)in_off_el * esz;
      size_t out_off_b = (size_t)lin * esz;
      std::memcpy(dst + out_off_b, src + in_off_b, esz);
    }

    return {Tensor::Raw(std::move(out))};
  }

  // 3) ScalarTensor handled above; anything else shouldn't reach
  throw std::runtime_error("vkcnn: Slice: unsupported data kind (internal)");
}

} // namespace vkcnn::details
