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
        // TODO: choose your preferred API to create a fresh symbol for this
        // label. Common patterns:
        //   s = state.symGraph->fresh(label);
        //   s = state.symGraph->new_symbol(label);
        // For now, we assume a `fresh(label)`-style API:
        s = state.symGraph->var();
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
    if (context == ValueInfoImportContext::Output)
      throw std::runtime_error(fmt::format(
          "vkcnn: output tensor \"{}\" does not define a type.", name));
    if (context == ValueInfoImportContext::Hint)
      return;
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
    if (context == ValueInfoImportContext::Hint)
      return;
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" missing tensor_type", name));
  }

  const auto &ttype = tp.tensor_type();

  const auto dtypeOpt = parse_data_type(ttype.elem_type());
  if (!dtypeOpt) {
    if (context == ValueInfoImportContext::Hint)
      return;
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" has unsupported elem_type {}", name,
                    data_type_to_string(ttype.elem_type())));
  }
  const Dtype dtype = *dtypeOpt;

  if (!ttype.has_shape()) {
    if (context == ValueInfoImportContext::Hint)
      return;
    throw std::runtime_error(fmt::format(
        "vkcnn: tensor {} has unknown shape (dynamic rank unsupported)", name));
  }

  // Convert ONNX shape -> TensorShape(Symbolic)
  TensorShape tshape =
      parse_vi_shape_to_tensor_shape(state, ttype.shape(), name);
  const size_t r = tshape.rank();

  // ---------- HINT ----------
  if (context == ValueInfoImportContext::Hint) {
    auto it = state.tensors.map.find(name);
    if (it == state.tensors.map.end())
      return; // nothing to refine

    Tensor &t = it->second;
    if (t.isDevice()) {
      // Optionally set device float type hint if needed
      // (Only if you want to honor dtype hints here)
      // maybe_set_device_float_type_from_dtype(const_cast<DeviceTensor&>(t.device()),
      // dtype); Shape refinements could go here if you want to cross-check.
    } else if (t.isHost()) {
      // Nothing critical for constants; you could cross-check shapes/dtypes if
      // desired.
    }
    return;
  }

  // ---------- INPUT ----------
  if (context == ValueInfoImportContext::Input) {
    // vkcnn expects exactly one runtime input and we represent it with
    // DeviceTensor
    if (r != 3 && r != 4) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Input \"{}\" must be rank 3 (CHW) or 4 (NCHW); got {}", name,
          r));
    }

    // NCHW (r==4) or CHW (r==3)
    size_t axC = (r == 4) ? 1 : 0;
    size_t axH = (r == 4) ? 2 : 1;
    size_t axW = (r == 4) ? 3 : 2;

    if (r == 4) {
      // If N is constant, require 1
      if (tshape[0].isConstant()) {
        if (tshape[0].constant() != 1) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Input tensor (\"{}\") has fixed batch {} (unsupported).",
              name, tshape[0].constant()));
        }
      } else {
        // N is symbolic (dim_param or unknown) → force to 1.
        // Runtime will select 1 anyway and not expose dynamic batch sizes.
        tshape[0] = Symbolic{state.symGraph, Sym::Const(1)};
      }
    }

    // Channels must be a known positive constant for the device handle
    // creation.
    const unsigned C =
        get_const_channels_or_throw(tshape, axC, "input channels");

    // Height/Width: pass as Sym (symbolic is fine)
    Sym Hs = static_cast<Sym>(*tshape[axH]); // Symbolic -> Sym
    Sym Ws = static_cast<Sym>(*tshape[axW]);

    // Optional float type hint from dtype
    std::optional<vkcnn::FloatType> hint = dtype_to_float_type(dtype);

    // Create runtime handle (vkcnn::Tensor). Keep your existing API:
    // channels, ??? (group/nullopt), float type hint, width, height
    vkcnn::Tensor rt = state.output.input(C, std::nullopt, hint, Ws, Hs);

    // Store as DeviceTensor
    DeviceTensor dev(r, std::move(rt));
    state.tensors.map.emplace(name, Tensor::Device(std::move(dev)));
    return;
  }

  // ---------- OUTPUT ----------
  // Must exist and be DeviceTensor (no constant outputs).
  auto it = state.tensors.map.find(name);
  if (it == state.tensors.map.end()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output \"{}\" was never produced by any node.", name));
  }
  const Tensor &produced = it->second;
  if (!produced.isDevice()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Output (\"{}\") is not a runtime tensor (constant "
                    "outputs unsupported).",
                    name));
  }

  DeviceTensor dev = produced.device();
  const auto prodShape = dev.shape(); // TensorShape
  const size_t pr = prodShape.rank();
  if (pr != 3 && pr != 4) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output tensor (\"{}\") must be rank 3 or 4; got {}.", name,
        pr));
  }

  // Interpret both as (N)CHW
  const size_t paxC = (pr == 4) ? 1 : 0;
  const size_t paxH = (pr == 4) ? 2 : 1;
  const size_t paxW = (pr == 4) ? 3 : 2;

  // Compare provided (tshape) to produced (prodShape) where concrete.
  // Channels (device channels are constant in your vkcnn::Tensor).
  {
    const unsigned rtC = dev.handle().channels();
    const auto &vd = tshape[paxC];
    if (vd.isConstant()) {
      if ((unsigned)vd.constant() != rtC) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Output (\"{}\") channel mismatch. Expected {}, got {}.",
            name, vd.constant(), rtC));
      }
    } else {
      // labeled/symbolic: if you track labels in symbolMap, you could check
      // here by resolving and comparing to rtC. Otherwise we accept symbolic
      // here.
      auto resolved = vd.resolve();
      if (resolved.isConstant() && (unsigned)resolved.constant() != rtC) {
        throw std::runtime_error(
            fmt::format("vkcnn: Output (\"{}\") channel mismatch.", name));
      }
    }
  }

  // Height/Width check: compare symbolically if possible
  auto check_axis = [&](size_t axProvided, size_t axProduced, const char *nm) {
    const auto &want = tshape[axProvided];
    const auto &got = prodShape[axProduced];
    if (want.isConstant() && got.isConstant()) {
      if (want.constant() != got.constant()) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Output (\"{}\") {} mismatch. Expected {}, got {}.", name,
            nm, want.constant(), got.constant()));
      }
    } else {
      // If symbolic, we can compare via == which resolves through the same
      // graph.
      if (!(want == got)) {
        // They may still be compatible but unresolved; be strict for now.
        throw std::runtime_error(fmt::format(
            "vkcnn: Output (\"{}\") {} symbolic mismatch.", name, nm));
      }
    }
  };
  check_axis(paxH, paxH, "height");
  check_axis(paxW, paxW, "width");

  // DType consistency (optional): if device type already set, compare; else
  // set.
  if (auto want = dtype_to_float_type(dtype)) {
    auto &h = dev.handle();
    if (h.type().has_value() && h.type().value() != *want) {
      throw std::runtime_error("vkcnn: Output tensor type mismatch.");
    }
    if (!h.type().has_value()) {
      h.setType(*want);
    }
  }

  // Finalize output into backend (same as before)
  state.output.output(dev.handle());
}

} // namespace vkcnn::details
