#include "frontend/onnx/details/import_value_info.hpp"
#include "frontend/onnx/details/values/Tensor.hpp"

#include <onnx.pb.h>

namespace denox::onnx::details {

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
  return static_cast<unsigned>(v);
}

void maybe_set_device_float_type_from_dtype(DeviceTensor &dev, Dtype onnxElem) {
  // If you have a helper
  // dtype_to_float_type(Dtype)->optional<vkcnn::FloatType>, use it here to set
  // dev.handle().setType(...)
  if (auto want = onnxElem.toDenoxType()) {
    // You mentioned handle().type() is optional and there is setType().
    // If your API needs non-const access:
    dev.handle().setType(*want);
  }
}

void import_value_info(
    ImportState &state, const ::onnx::ValueInfoProto &valueInfo,
    ValueInfoImportContext context) {
  const memory::string &name = valueInfo.name();
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
  memory::optional<Dtype> dtypeOpt = Dtype::parse(ttype.elem_type());
  if (!dtypeOpt) {
    if (context == ValueInfoImportContext::Input) {
      throw std::runtime_error(
          fmt::format("vkcnn: Tensor \"{}\" has unsupported elem_type {}", name,
                      Dtype::parse_to_string(ttype.elem_type())));
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

    TensorShape tshape = TensorShape::parse(ttype.shape(), state.symGraph);
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
        tshape[0] = compiler::Symbolic{state.symGraph, compiler::Sym::Const(1)};
      }
    }

    // Channels must be positive constant to build the runtime handle
    const unsigned C =
        get_const_channels_or_throw(tshape, axC, "input channels");

    // Height/Width may be symbolic
    const compiler::Sym Hs = static_cast<compiler::Sym>(*tshape[axH]);
    const compiler::Sym Ws = static_cast<compiler::Sym>(*tshape[axW]);

    // Optional float-type hint from dtype
    memory::optional<memory::Dtype> hint =
        dtypeOpt ? dtypeOpt->toDenoxType() : memory::nullopt;

    compiler::Tensor rt = state.output.input(C, memory::nullopt, hint, Ws, Hs);

    DeviceTensor dev(r, std::move(rt));
    state.tensors.emplace(name, Tensor::Device(std::move(dev)));
    return;
  }

  // ---------------- OUTPUT ----------------
  // Must exist and be DeviceTensor
  auto it = state.tensors.find(name);
  if (it == state.tensors.end()) {
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
    if (auto want = dtypeOpt->toDenoxType()) {
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

} // namespace denox::onnx::details
