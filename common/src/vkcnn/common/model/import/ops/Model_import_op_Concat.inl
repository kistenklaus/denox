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

static std::vector<Tensor> import_op_Concat_runtime(
    [[maybe_unused]] ImportState &state,
    [[maybe_unused]] std::span<const std::optional<Tensor>> inputs,
    [[maybe_unused]] std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {
  throw std::runtime_error(fmt::format(
      "vkcnn: operation Concat is not supported (node = \"{}\")", node.name()));
}

static std::vector<Tensor>
import_op_Concat(ImportState &state,
                 std::span<const std::optional<Tensor>> inputs,
                 std::size_t outputCount,
                 const std::unordered_map<std::string, Tensor> &attributes,
                 opset_version version, const onnx::NodeProto &node) {

  // ---- contract
  if (outputCount != 1) {
    throw std::runtime_error("vkcnn: Concat must produce exactly one output");
  }
  if (inputs.empty()) {
    throw std::runtime_error("vkcnn: Concat requires at least one input");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value()) {
      throw std::runtime_error("vkcnn: Concat input present but not provided");
    }
  }

  // ---- redirect if any RuntimeTensor
  for (const auto &opt : inputs) {
    if (opt->isRuntimeTensor()) {
      // Delegate to runtime variant
      return import_op_Concat_runtime(state, inputs, outputCount, attributes,
                                      version, node);
    }
  }

  // ---- parse 'axis' (required by ONNX Concat)
  auto get_axis = [&]() -> int64_t {
    auto it = attributes.find("axis");
    if (it == attributes.end()) {
      throw std::runtime_error("vkcnn: Concat requires attribute 'axis'");
    }
    const Tensor &a = it->second;
    if (a.isUnknown())
      return 0; // your policy: present-but-unknown => 0
    if (!a.isScalar())
      throw std::runtime_error(
          "vkcnn: Concat attribute 'axis' must be an integer scalar");
    const auto &s = a.scalar();
    if (s.dtype == Dtype::Int64 || s.dtype == Dtype::Int32 ||
        s.dtype == Dtype::Int16)
      return (int64_t)s.v.i;
    if (s.dtype == Dtype::Uint64 || s.dtype == Dtype::Uint32 ||
        s.dtype == Dtype::Uint16 || s.dtype == Dtype::Uint8)
      return (int64_t)s.v.u;
    throw std::runtime_error("vkcnn: Concat attribute 'axis' must be integer");
  };
  int64_t axis_attr = get_axis();

  // ---- classify inputs
  auto is_scalar_like = [](const Tensor &t) -> bool {
    if (t.isScalar())
      return true;
    if (t.isRaw() && t.raw().shape.isScalar())
      return true; // 0-D Raw
    return false;
  };

  bool any_shape = false, all_shape = true;
  bool any_raw_rank_ge1 = false, all_raw_rank_ge1 = true;
  bool any_scalar_like = false, all_scalar_like = true;
  bool any_raw = false, any_scalar = false, any_string = false;

  for (const auto &opt : inputs) {
    const Tensor &t = *opt;
    any_shape |= t.isShape();
    all_shape &= t.isShape();

    bool is_raw0 = t.isRaw() && t.raw().shape.isScalar();
    bool is_rawN = t.isRaw() && !t.raw().shape.isScalar();
    any_raw_rank_ge1 |= is_rawN;
    all_raw_rank_ge1 &= is_rawN || inputs.size() == 1; // will refine below

    bool is_slike = is_scalar_like(t);
    any_scalar_like |= is_slike;
    all_scalar_like &= is_slike;

    any_raw |= t.isRaw();
    any_scalar |= t.isScalar();
    any_string |= t.isString();
  }

  if (any_string) {
    throw std::runtime_error(
        "vkcnn: Concat on string tensors is not supported");
  }

  // ========================= CASE 1: all ShapeTensor =======================
  if (all_shape) {
    // Each ShapeTensor should be a 1-D vector of Dims (not scalar)
    for (const auto &opt : inputs) {
      if (opt->shapeTensor().isScalar()) {
        throw std::runtime_error(
            "vkcnn: Concat on scalar ShapeTensor is invalid");
      }
    }
    // rank = 1 for these vectors -> only axis 0 (or -1 normalized)
    int64_t ax = axis_attr;
    if (ax < 0)
      ax += 1;
    if (ax != 0) {
      throw std::runtime_error(
          "vkcnn: Concat on ShapeTensor only supports axis 0");
    }

    ShapeVector out;
    for (const auto &opt : inputs) {
      const auto &v = opt->shapeTensor().dims();
      for (const Dim &d : v)
        out.push_back(d);
    }
    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // ===================== CASE 2: all scalar-like (Scalar or Raw 0-D) =======
  if (all_scalar_like) {
    // Only axis 0 makes sense for a list of scalars -> normalized axis must be
    // 0
    int64_t ax = axis_attr;
    if (ax < 0)
      ax += 1;
    if (ax != 0) {
      throw std::runtime_error("vkcnn: Concat of scalars only supports axis 0");
    }

    // Readers
    auto as_i64_scalar = [&](const Tensor &t) -> std::optional<int64_t> {
      if (t.isScalar()) {
        const auto &s = t.scalar();
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
                "vkcnn: Concat: uint64 scalar too large to fit int64");
          return (int64_t)s.v.u;
        case Dtype::Uint32:
        case Dtype::Uint16:
        case Dtype::Uint8:
          return (int64_t)s.v.u;
        default:
          return std::nullopt;
        }
      }
      if (t.isRaw()) {
        const auto &rt = t.raw();
        if (!rt.shape.isScalar())
          return std::nullopt;
        auto need = [&](size_t n) {
          if (rt.raw.size() != n)
            throw std::runtime_error("vkcnn: Concat raw scalar size mismatch");
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
            throw std::runtime_error("vkcnn: Concat: raw uint64 too large");
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
      }
      return std::nullopt;
    };
    auto as_f_scalar = [&](const Tensor &t) -> std::optional<double> {
      if (t.isScalar()) {
        const auto &s = t.scalar();
        if (s.dtype == Dtype::Float64)
          return (double)s.v.float64;
        if (s.dtype == Dtype::Float32)
          return (double)s.v.float32;
        return std::nullopt;
      }
      if (t.isRaw()) {
        const auto &rt = t.raw();
        if (!rt.shape.isScalar())
          return std::nullopt;
        if (rt.type == Dtype::Float64) {
          if (rt.raw.size() != sizeof(f64))
            throw std::runtime_error("vkcnn: Concat raw f64 size mismatch");
          f64 v;
          std::memcpy(&v, rt.raw.data(), sizeof v);
          return (double)v;
        }
        if (rt.type == Dtype::Float32) {
          if (rt.raw.size() != sizeof(f32))
            throw std::runtime_error("vkcnn: Concat raw f32 size mismatch");
          f32 v;
          std::memcpy(&v, rt.raw.data(), sizeof v);
          return (double)v;
        }
        return std::nullopt;
      }
      return std::nullopt;
    };

    bool any_float = false, any_f64 = false;
    for (const auto &opt : inputs) {
      const Tensor &t = *opt;
      if (as_f_scalar(t).has_value()) {
        any_float = true;
        if (t.isScalar() && t.scalar().dtype == Dtype::Float64)
          any_f64 = true;
        if (t.isRaw() && t.raw().type == Dtype::Float64)
          any_f64 = true;
      }
    }

    RawTensor out;
    out.shape = ShapeTensor::Vec(inputs.size()); // [N]
    if (any_float) {
      out.type = any_f64 ? Dtype::Float64 : Dtype::Float32;
      const size_t esz = dtype_size(out.type);
      out.raw.resize(inputs.size() * esz);
      size_t off = 0;
      for (const auto &opt : inputs) {
        double v;
        if (auto fv = as_f_scalar(*opt))
          v = *fv;
        else if (auto iv = as_i64_scalar(*opt))
          v = (double)*iv;
        else
          throw std::runtime_error(
              "vkcnn: Concat scalar list contains non-numeric");
        if (out.type == Dtype::Float64) {
          f64 d = (f64)v;
          std::memcpy(out.raw.data() + off, &d, sizeof d);
          off += sizeof d;
        } else {
          f32 f = (f32)v;
          std::memcpy(out.raw.data() + off, &f, sizeof f);
          off += sizeof f;
        }
      }
    } else {
      out.type = Dtype::Int64;
      const size_t esz = dtype_size(out.type);
      out.raw.resize(inputs.size() * esz);
      size_t off = 0;
      for (const auto &opt : inputs) {
        auto iv = as_i64_scalar(*opt);
        if (!iv)
          throw std::runtime_error(
              "vkcnn: Concat scalar list contains non-integer");
        int64_t v = *iv;
        std::memcpy(out.raw.data() + off, &v, sizeof v);
        off += sizeof v;
      }
    }
    return {Tensor::Raw(std::move(out))};
  }

  // ===================== CASE 3: all RawTensor (rank >= 1) =================
  if (any_raw_rank_ge1 && !any_shape && !any_scalar_like) {
    struct RtInfo {
      const RawTensor *rt;
      std::vector<int64_t> sizes; // static sizes, length = rank
    };
    std::vector<RtInfo> rts;
    rts.reserve(inputs.size());

    // Init from first non-scalar RawTensor
    size_t first_idx = 0;
    while (first_idx < inputs.size() &&
           !(inputs[first_idx]->isRaw() &&
             !inputs[first_idx]->raw().shape.isScalar()))
      ++first_idx;
    if (first_idx == inputs.size()) {
      throw std::logic_error("vkcnn: Concat internal classification error");
    }
    const RawTensor &rt0 = inputs[first_idx]->raw();
    if (rt0.type == Dtype::String) {
      throw std::runtime_error(
          "vkcnn: Concat on RawTensor<string> not supported");
    }
    int64_t rank = rt0.shape.isScalar() ? 0 : (int64_t)rt0.shape.dims().size();
    if (rank == 0) {
      throw std::runtime_error(
          "vkcnn: Concat on rank-0 RawTensors not supported (axis invalid)");
    }
    auto read_static_sizes = [&](const RawTensor &rt) -> std::vector<int64_t> {
      std::vector<int64_t> sz;
      sz.reserve((size_t)rank);
      for (const Dim &d : rt.shape.dims()) {
        if (!d.isConst())
          throw std::runtime_error(
              "vkcnn: Concat RawTensor requires static shapes");
        sz.push_back((int64_t)d.value());
      }
      return sz;
    };
    std::vector<int64_t> base_sizes = read_static_sizes(rt0);
    Dtype dtype = rt0.type;

    rts.push_back(RtInfo{&rt0, base_sizes});

    // Collect/validate others (skip scalar-likes; there shouldn't be any here)
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i == first_idx)
        continue;
      const Tensor &t = *inputs[i];
      if (!(t.isRaw() && !t.raw().shape.isScalar())) {
        throw std::runtime_error(
            "vkcnn: Concat mixing higher-rank RawTensors with scalars or other "
            "kinds is not supported");
      }
      const RawTensor &rt = t.raw();
      if (rt.type == Dtype::String)
        throw std::runtime_error(
            "vkcnn: Concat on RawTensor<string> not supported");
      int64_t r = (int64_t)rt.shape.dims().size();
      if (r != rank)
        throw std::runtime_error(
            "vkcnn: Concat inputs must have the same rank");
      if (rt.type != dtype)
        throw std::runtime_error(
            "vkcnn: Concat inputs must have the same dtype");
      rts.push_back(RtInfo{&rt, read_static_sizes(rt)});
    }

    // Normalize axis
    int64_t ax = axis_attr;
    if (ax < 0)
      ax += rank;
    if (ax < 0 || ax >= rank) {
      throw std::runtime_error(
          "vkcnn: Concat 'axis' out of range for input rank");
    }

    // Non-axis dims must match
    for (size_t i = 1; i < rts.size(); ++i) {
      for (int64_t d = 0; d < rank; ++d) {
        if (d == ax)
          continue;
        if (rts[i].sizes[(size_t)d] != rts[0].sizes[(size_t)d])
          throw std::runtime_error(
              "vkcnn: Concat dimension mismatch on non-axis dims");
      }
    }

    // Sizing helpers
    auto prod = [](const std::vector<int64_t> &v) -> int64_t {
      int64_t p = 1;
      for (auto x : v)
        p *= x;
      return p;
    };
    auto prod_range = [](const std::vector<int64_t> &v, size_t a,
                         size_t b) -> int64_t {
      int64_t p = 1;
      for (size_t i = a; i < b; ++i)
        p *= v[i];
      return p;
    };

    // Output axis size and raw size sanity
    int64_t out_axis = 0;
    const size_t esize = dtype_size(dtype);
    for (const auto &ri : rts) {
      out_axis += ri.sizes[(size_t)ax];
      int64_t elems = prod(ri.sizes);
      if (ri.rt->raw.size() != (size_t)elems * esize)
        throw std::runtime_error("vkcnn: Concat raw payload size mismatch");
    }

    // Build output shape
    ShapeVector out_dims;
    out_dims.reserve((size_t)rank);
    for (int64_t d = 0; d < rank; ++d) {
      if (d == ax)
        out_dims.push_back(Dim::Const((uint64_t)out_axis));
      else
        out_dims.push_back(Dim::Const((uint64_t)rts[0].sizes[(size_t)d]));
    }

    // Copy bytes: for each outer block, append all inputs' axis-chunks
    int64_t inner = prod_range(rts[0].sizes, (size_t)(ax + 1), (size_t)rank);
    int64_t outer = prod_range(rts[0].sizes, 0, (size_t)ax);

    RawTensor out;
    out.type = dtype;
    out.shape = ShapeTensor::Tensor(std::move(out_dims));
    out.raw.resize((size_t)outer * (size_t)out_axis * (size_t)inner * esize);

    uint8_t *dst = reinterpret_cast<uint8_t *>(out.raw.data());
    size_t dst_off = 0;

    for (int64_t ob = 0; ob < outer; ++ob) {
      for (const auto &ri : rts) {
        size_t copy_bytes =
            (size_t)ri.sizes[(size_t)ax] * (size_t)inner * esize;
        size_t stride_bytes = (size_t)ri.sizes[(size_t)ax] * (size_t)inner *
                              esize; // per-outer stride
        const uint8_t *src =
            reinterpret_cast<const uint8_t *>(ri.rt->raw.data());
        size_t src_base = (size_t)ob * stride_bytes;
        std::memcpy(dst + dst_off, src + src_base, copy_bytes);
        dst_off += copy_bytes;
      }
    }

    return {Tensor::Raw(std::move(out))};
  }

  // ... keep the earlier cases unchanged ...

  // ===== NEW: allow mixing ShapeTensor with integer scalars / 1-D Raw(int)
  // ====
  if (any_shape) {
    // Normalize axis for a 1-D shape vector result
    int64_t ax = axis_attr;
    if (ax < 0)
      ax += 1;
    if (ax != 0) {
      throw std::runtime_error(
          "vkcnn: Concat(ShapeTensor, …) only supports axis 0");
    }

    // Build a ShapeVector by walking inputs in order
    ShapeVector out;

    auto push_i64_as_dim = [&](int64_t v) {
      if (v < 0) {
        throw std::runtime_error(
            "vkcnn: negative value in Concat -> ShapeTensor not representable");
      }
      out.push_back(Dim::Const(static_cast<uint64_t>(v)));
    };

    auto push_raw_1d_ints = [&](const RawTensor &rt) {
      // Accept Int{8,16,32,64}/UInt{8,16,32,64}; shape must be [N] (1-D)
      if (!rt.shape.isTensor() || rt.shape.dims().size() != 1)
        throw std::runtime_error(
            "vkcnn: Concat mixing Shape requires 1-D Raw int tensor");
      const Dim &d0 = rt.shape.dims()[0];
      if (!d0.isConst())
        throw std::runtime_error(
            "vkcnn: Concat mixing Shape requires static length");
      size_t n = static_cast<size_t>(d0.value());

      auto copy_vals = [&](auto *ptr) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        const T *p = reinterpret_cast<const T *>(rt.raw.data());
        size_t cnt = rt.raw.size() / sizeof(T);
        if (cnt != n)
          throw std::runtime_error("vkcnn: Concat raw payload size mismatch");
        for (size_t i = 0; i < n; ++i)
          push_i64_as_dim(static_cast<int64_t>(p[i]));
      };

      switch (rt.type) {
      case Dtype::Int64:
        copy_vals((int64_t *)nullptr);
        break;
      case Dtype::Int32:
        copy_vals((int32_t *)nullptr);
        break;
      case Dtype::Int16:
        copy_vals((int16_t *)nullptr);
        break;
      case Dtype::Int8:
        copy_vals((int8_t *)nullptr);
        break;
      case Dtype::Uint64:
        copy_vals((uint64_t *)nullptr);
        break;
      case Dtype::Uint32:
        copy_vals((uint32_t *)nullptr);
        break;
      case Dtype::Uint16:
        copy_vals((uint16_t *)nullptr);
        break;
      case Dtype::Uint8:
        copy_vals((uint8_t *)nullptr);
        break;
      default:
        throw std::runtime_error(
            "vkcnn: Concat mixing Shape requires integer Raw dtype");
      }
    };

    for (const auto &opt : inputs) {
      const Tensor &t = *opt;

      if (t.isShape()) {
        const auto &sv = t.shapeTensor();
        if (sv.isScalar())
          throw std::runtime_error(
              "vkcnn: Concat on scalar ShapeTensor is invalid");
        for (const Dim &d : sv.dims())
          out.push_back(d);
        continue;
      }

      // Accept integer scalars (ScalarTensor or Raw 0-D) and append as one dim
      auto try_scalar_i64 = [&](const Tensor &s) -> bool {
        // ScalarTensor?
        if (s.isScalar()) {
          const auto &sc = s.scalar();
          switch (sc.dtype) {
          case Dtype::Int64:
            push_i64_as_dim(sc.v.i);
            return true;
          case Dtype::Int32:
          case Dtype::Int16:
          case Dtype::Int8:
            push_i64_as_dim((int64_t)sc.v.i);
            return true;
          case Dtype::Uint64:
            if (sc.v.u > (uint64_t)std::numeric_limits<int64_t>::max())
              throw std::runtime_error("vkcnn: Concat integer too large");
            push_i64_as_dim((int64_t)sc.v.u);
            return true;
          case Dtype::Uint32:
          case Dtype::Uint16:
          case Dtype::Uint8:
            push_i64_as_dim((int64_t)sc.v.u);
            return true;
          default:
            return false;
          }
        }
        // Raw 0-D?
        if (s.isRaw() && s.raw().shape.isScalar()) {
          const auto &rt = s.raw();
          auto need = [&](size_t n) {
            if (rt.raw.size() != n)
              throw std::runtime_error(
                  "vkcnn: Concat raw scalar size mismatch");
          };
          switch (rt.type) {
          case Dtype::Int64: {
            need(8);
            int64_t v;
            std::memcpy(&v, rt.raw.data(), 8);
            push_i64_as_dim(v);
            return true;
          }
          case Dtype::Int32: {
            need(4);
            int32_t v;
            std::memcpy(&v, rt.raw.data(), 4);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Int16: {
            need(2);
            int16_t v;
            std::memcpy(&v, rt.raw.data(), 2);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Int8: {
            need(1);
            int8_t v;
            std::memcpy(&v, rt.raw.data(), 1);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Uint64: {
            need(8);
            uint64_t v;
            std::memcpy(&v, rt.raw.data(), 8);
            if (v > (uint64_t)std::numeric_limits<int64_t>::max())
              throw std::runtime_error("vkcnn: Concat integer too large");
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Uint32: {
            need(4);
            uint32_t v;
            std::memcpy(&v, rt.raw.data(), 4);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Uint16: {
            need(2);
            uint16_t v;
            std::memcpy(&v, rt.raw.data(), 2);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          case Dtype::Uint8: {
            need(1);
            uint8_t v;
            std::memcpy(&v, rt.raw.data(), 1);
            push_i64_as_dim((int64_t)v);
            return true;
          }
          default:
            return false;
          }
        }
        return false;
      };

      if (try_scalar_i64(t))
        continue;

      // Accept 1-D Raw ints vector
      if (t.isRaw() && !t.raw().shape.isScalar()) {
        push_raw_1d_ints(t.raw());
        continue;
      }

      // Anything else is not supported in the Shape-mixing path
      throw std::runtime_error("vkcnn: Concat mixing ShapeTensor with "
                               "non-integer tensors is not supported");
    }

    // ===================== unsupported mixes ================================

    return {Tensor::Shape(ShapeTensor::Tensor(std::move(out)))};
  }

  // (otherwise keep the remaining "unsupported mixes" throws)
  if (any_raw || any_scalar) {
    throw std::runtime_error("vkcnn: Concat mixing higher-rank RawTensors with "
                             "scalars is not supported");
  }

  throw std::runtime_error(fmt::format(
      "vkcnn: Concat unsupported input combination (node='{}')", node.name()));
}

} // namespace vkcnn::details
