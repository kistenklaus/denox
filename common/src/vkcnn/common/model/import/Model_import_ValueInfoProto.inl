#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/symbolic/Sym.hpp"
#include "vkcnn/common/symbolic/Symbolic.hpp"
#include <fmt/format.h>
#include <onnx.pb.h>
#include <stdexcept>

namespace vkcnn::details {

enum ValueInfoImportContext {
  Input,
  Output,
  Hint,
};

// Helpers (file-local)

static TensorShape
parse_vi_shape_to_tensor_shape(ImportState &state,
                               const onnx::TensorShapeProto &shp,
                               const std::string &tensorName) {

  std::vector<Symbolic> dims;
  dims.reserve((size_t)shp.dim_size());
  auto &g = state.symGraph;
  if (!g)
    throw std::runtime_error("vkcnn: symGraph is null");

  for (int i = 0; i < shp.dim_size(); ++i) {
    const auto &d = shp.dim(i);
    if (d.has_dim_value()) {
      const int64_t v = d.dim_value();
      if (v < 0) {
        throw std::runtime_error(fmt::format(
            "vkcnn: {} has negative dim at axis {}", tensorName, i));
      }
      dims.emplace_back(g, Sym::Const(v));
    } else if (d.has_dim_param()) {
      const std::string &label = d.dim_param();
      if (label.empty()) {
        throw std::runtime_error(fmt::format(
            "vkcnn: {} has empty dim_param at axis {}", tensorName, i));
      }
      auto it = state.symbolMap.find(label);
      Sym s = Sym::Const(0);
      if (it != state.symbolMap.end()) {
        s = it->second;
      } else {
        if (shp.dim_size() == 4 && i == 0) {
          s = Sym::Const(1);
        } else {
          s = state.symGraph->var();
        }
        state.symbolMap.emplace(label, s);
      }
      dims.emplace_back(g, s);
    } else {
      // Completely unknown dim (neither value nor param) — give it a fresh
      // symbol.
      const std::string autoLabel = tensorName + "/dim" + std::to_string(i);
      Sym s = state.symGraph->var();
      state.symbolMap.emplace(autoLabel, s);
      dims.emplace_back(g, s);
    }
  }

  return TensorShape{g, std::move(dims)};
}

static unsigned get_const_channels_or_throw(const TensorShape &s, size_t axis,
                                            const char *what) {
  const auto &d = s[axis];
  if (!d.isConstant()) {
    throw std::runtime_error(
        fmt::format("vkcnn: {} must be a constant (axis {})", what, axis));
  }
  const auto v = d.constant();
  if (v <= 0) {
    throw std::runtime_error(
        fmt::format("vkcnn: {} must be positive, got {}", what, v));
  }
  return (unsigned)v;
}

static void maybe_set_device_float_type_from_dtype(DeviceTensor &dev,
                                                   Dtype onnxElem) {
  // If you have a helper
  // dtype_to_float_type(Dtype)->optional<vkcnn::FloatType>, use it here to set
  // dev.handle().setType(...)
  if (auto want = dtype_to_float_type(onnxElem)) {
    // You mentioned handle().type() is optional and there is setType().
    // If your API needs non-const access:
    dev.handle().setType(*want);
  }
}

// Main function

static void import_value_info(ImportState &state,
                              const onnx::ValueInfoProto &valueInfo,
                              ValueInfoImportContext context) {
  const std::string &name = valueInfo.name();
  if (name.empty())
    throw std::runtime_error("vkcnn: \"\" is not a valid tensor name.");

  if (!valueInfo.has_type()) {
    if (context == ValueInfoImportContext::Input)
      throw std::runtime_error(fmt::format(
          "vkcnn: input tensor \"{}\" does not define a type.", name));
    if (context == ValueInfoImportContext::Output ||
        context == ValueInfoImportContext::Hint)
      return; // ignore missing type for Output/Hint
    throw std::logic_error("Unexpected ValueInfoImportContext.");
  }

  const auto &tp = valueInfo.type();
  if (tp.has_optional_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported optional_type", name));
  if (tp.has_sparse_tensor_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sparse_tensor_type", name));
  if (tp.has_map_type())
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" has unsupported map_type", name));
  if (tp.has_sequence_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sequence_type", name));
  if (!tp.has_tensor_type()) {
    if (context == ValueInfoImportContext::Hint ||
        context == ValueInfoImportContext::Output)
      return; // ignore for Hint/Output
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" missing tensor_type", name));
  }

  const auto &ttype = tp.tensor_type();

  // dtype is optional for Output/Hint (we only use it to set/verify float type)
  std::optional<Dtype> dtypeOpt = parse_data_type(ttype.elem_type());
  if (!dtypeOpt) {
    if (context == ValueInfoImportContext::Input) {
      throw std::runtime_error(
          fmt::format("vkcnn: Tensor \"{}\" has unsupported elem_type {}", name,
                      data_type_to_string(ttype.elem_type())));
    }
    // For Output/Hint we just skip dtype-based checks.
  }

  // ---------------- HINT ----------------
  if (context == ValueInfoImportContext::Hint) {
    // We don't enforce anything here.
    return;
  }

  // ---------------- INPUT ----------------
  if (context == ValueInfoImportContext::Input) {
    if (!ttype.has_shape()) {
      throw std::runtime_error(fmt::format(
          "vkcnn: tensor {} has unknown shape (dynamic rank unsupported)",
          name));
    }

    TensorShape tshape =
        parse_vi_shape_to_tensor_shape(state, ttype.shape(), name);
    const size_t r = tshape.rank();

    if (r != 3 && r != 4) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Input \"{}\" must be rank 3 (CHW) or 4 (NCHW); got {}", name,
          r));
    }

    const size_t axC = (r == 4) ? 1 : 0;
    const size_t axH = (r == 4) ? 2 : 1;
    const size_t axW = (r == 4) ? 3 : 2;

    if (r == 4) {
      // If N is constant, require 1; else force to 1.
      if (tshape[0].isConstant()) {
        if (tshape[0].constant() != 1) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Input tensor (\"{}\") has fixed batch {} (unsupported).",
              name, tshape[0].constant()));
        }
      } else {
        tshape[0] = Symbolic{state.symGraph, Sym::Const(1)};
      }
    }

    // Channels must be positive constant to build the runtime handle
    const unsigned C =
        get_const_channels_or_throw(tshape, axC, "input channels");

    // Height/Width may be symbolic
    const Sym Hs = static_cast<Sym>(*tshape[axH]);
    const Sym Ws = static_cast<Sym>(*tshape[axW]);

    // Optional float-type hint from dtype
    std::optional<vkcnn::FloatType> hint =
        dtypeOpt ? dtype_to_float_type(*dtypeOpt) : std::nullopt;

    vkcnn::Tensor rt = state.output.input(C, std::nullopt, hint, Ws, Hs);

    DeviceTensor dev(r, std::move(rt));
    state.tensors.map.emplace(name, Tensor::Device(std::move(dev)));
    return;
  }

  // ---------------- OUTPUT ----------------
  // Must exist and be DeviceTensor
  auto it = state.tensors.map.find(name);
  if (it == state.tensors.map.end()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output \"{}\" was never produced by any node.", name));
  }
  if (!it->second.isDevice()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Output (\"{}\") is not a runtime tensor (constant "
                    "outputs unsupported).",
                    name));
  }

  DeviceTensor dev = it->second.device();

  // (A) DType consistency: if ONNX dtype is supported → verify or set
  if (dtypeOpt) {
    if (auto want = dtype_to_float_type(*dtypeOpt)) {
      auto &h = dev.handle();
      if (h.type().has_value() && h.type().value() != *want) {
        throw std::runtime_error("vkcnn: Output tensor type mismatch.");
      }
      if (!h.type().has_value()) {
        h.setType(*want);
      }
    }
  }

  // (B) Channel-count check only (ignore all other dims/symbols)
  // We only do this if ONNX provided a tensor shape and a concrete channel dim.
  if (ttype.has_shape()) {
    const auto &oshape = ttype.shape();

    // Determine which axis is channels from the produced tensor rank (3/4).
    const size_t pr = dev.shape().rank();
    if (pr != 3 && pr != 4) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Output tensor (\"{}\") must be rank 3 or 4; got {}.", name,
          pr));
    }
    const size_t chAxProduced = (pr == 4) ? 1 : 0;

    // If ONNX rank doesn't match or is weird, we don't try to reconcile; we
    // simply skip the channel check.
    const size_t orank = static_cast<size_t>(oshape.dim_size());
    if ((pr == 3 && orank == 3) || (pr == 4 && orank == 4)) {
      const size_t chAxOnnx = (orank == 4) ? 1 : 0;
      const auto &d = oshape.dim(static_cast<int>(chAxOnnx));
      if (d.has_dim_value()) {
        const int64_t v = d.dim_value();
        if (v < 0) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Output (\"{}\") has negative channel dim {}", name, v));
        }
        const unsigned rtC = dev.handle().channels();
        if (static_cast<unsigned>(v) != rtC) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Output (\"{}\") channel mismatch. Expected {}, got {}.",
              name, v, rtC));
        }
      }
      // If dim_param or unknown → we ignore (no symbol tracking here).
    }
  }

  // Finalize output in backend
  state.output.output(dev.handle());
}

} // namespace vkcnn::details
