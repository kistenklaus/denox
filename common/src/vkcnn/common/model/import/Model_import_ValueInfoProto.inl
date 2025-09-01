#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/symbolic/Sym.hpp"
#include "vkcnn/common/symbolic/Symbolic.hpp"
#include <fmt/format.h>
#include <limits>
#include <onnx.pb.h>
#include <stdexcept>

namespace vkcnn::details {

enum ValueInfoImportContext {
  Input,
  Output,
  Hint,
};

// ==== small IR for parsing ValueInfo shapes ====
struct RawDim {
  enum class Kind { Value, Label, Unknown } kind = Kind::Unknown;
  uint64_t value = 0; // valid when kind==Value
  std::string label;  // valid when kind==Label
};

static std::vector<RawDim> parse_raw_dims(const onnx::TensorShapeProto &shp,
                                          const std::string &name) {
  const int rank = shp.dim_size();
  if (rank == 0) {
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" must not be scalar (rank=0).", name));
  }
  std::vector<RawDim> out;
  out.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    const auto &d = shp.dim(i);
    if (d.has_dim_value()) {
      int64_t v = d.dim_value();
      if (v < 0) {
        throw std::runtime_error(
            fmt::format("vkcnn: Tensor \"{}\" has negative dim {}", name, v));
      }
      out.push_back(RawDim{RawDim::Kind::Value, static_cast<uint64_t>(v), {}});
    } else if (d.has_dim_param()) {
      out.push_back(RawDim{RawDim::Kind::Label, 0u, d.dim_param()});
    } else {
      // truly unknown
      out.push_back(RawDim{RawDim::Kind::Unknown, 0u, {}});
    }
  }
  return out;
}

struct CHWView {
  // For CHW (rank=3): N is empty
  // For NCHW (rank=4): N has a value
  const RawDim *N = nullptr;
  const RawDim *C = nullptr;
  const RawDim *H = nullptr;
  const RawDim *W = nullptr;
};

static CHWView chw_view_from_raw_dims(const std::vector<RawDim> &dims,
                                      const std::string &name, bool is_input) {
  CHWView v;
  if (dims.size() == 3) {
    v.C = &dims[0];
    v.H = &dims[1];
    v.W = &dims[2];
    return v;
  }
  if (dims.size() == 4) {
    v.N = &dims[0];
    v.C = &dims[1];
    v.H = &dims[2];
    v.W = &dims[3];
    return v;
  }
  throw std::runtime_error(fmt::format(
      "vkcnn: {} tensor (\"{}\") must have rank 3 (CHW) or 4 (NCHW), got {}.",
      is_input ? "Input" : "Output", name, dims.size()));
}

static Sym require_sym_with_label(ImportState &st, const std::string &label) {
  auto it = st.symbolMap.find(label);
  if (it != st.symbolMap.end())
    return it->second;
  Sym s = st.symGraph->var();
  st.symbolMap.emplace(label, s);
  return s;
}

static Dim make_input_dim(ImportState &st, const RawDim &rd) {
  switch (rd.kind) {
  case RawDim::Kind::Value:
    return Dim::Const(rd.value);
  case RawDim::Kind::Label:
    return Dim::Symbol(require_sym_with_label(st, rd.label));
  case RawDim::Kind::Unknown: {
    Sym s = st.symGraph->var();
    return Dim::Symbol(s);
  }
  }
  throw std::logic_error("unreachable");
}

struct DimEval {
  std::optional<uint64_t> c;
  std::optional<Sym> s;
};

static DimEval eval_dim(const Dim &d, SymGraph &g) {
  if (d.isConst())
    return {d.value(), std::nullopt};
  auto r = g.resolve(d.sym());
  if (r.isConstant())
    return {r.constant(), std::nullopt};
  return {std::nullopt, r};
}

static unsigned get_const_channels(const Dim &cdim, const std::string &name,
                                   ImportState &st) {
  DimEval e = eval_dim(cdim, *st.symGraph);
  if (!e.c)
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor (\"{}\") must have constant channel count.", name));
  uint64_t c64 = *e.c;
  if (c64 > std::numeric_limits<unsigned>::max())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor (\"{}\") channel count too large: {}", name, c64));
  return static_cast<unsigned>(c64);
}

static void check_output_axis(const RawDim &rd, const Symbolic &runtime_axis,
                              ImportState &st, std::string_view what,
                              const std::string &tensor_name) {
  if (rd.kind == RawDim::Kind::Unknown)
    return;
  if (rd.kind == RawDim::Kind::Value) {
    if (!runtime_axis.isConstant() ||
        static_cast<std::uint64_t>(runtime_axis.constant()) != rd.value) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Output (\"{}\"): {} mismatch. Expected {}, got {}.",
          tensor_name, what, rd.value,
          runtime_axis.isConstant() ? std::to_string(runtime_axis.constant())
                                    : std::string("<symbolic>")));
    }
    return;
  }
  const auto it = st.symbolMap.find(rd.label);
  if (it == st.symbolMap.end())
    throw std::runtime_error(fmt::format(
        "vkcnn: Output (\"{}\"): dim label \"{}\" not seen in inputs.",
        tensor_name, rd.label));
  const Sym expected = it->second;
  if (!runtime_axis.isSymbolic() || !(runtime_axis == expected))
    throw std::runtime_error(
        fmt::format("vkcnn: Output (\"{}\"): {} symbolic mismatch (label {}).",
                    tensor_name, what, rd.label));
}

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

  std::optional<Dtype> dtypeOpt = parse_data_type(ttype.elem_type());
  if (!dtypeOpt.has_value()) {
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

  const auto raw = parse_raw_dims(ttype.shape(), name);
  const bool is_input = (context == ValueInfoImportContext::Input);
  const CHWView view = chw_view_from_raw_dims(raw, name, is_input);

  if (context == ValueInfoImportContext::Hint) {
    auto it = state.tensors.map.find(name);
    if (it != state.tensors.map.end() && it->second.isRuntimeTensor()) {
      auto want = dtype_to_float_type(dtype);
      if (want && !it->second.runtime().tensor.type().has_value()) {
        it->second.runtime().tensor.setType(*want);
      }
    }
    return;
  }

  if (context == ValueInfoImportContext::Input) {
    ShapeVector dims;
    dims.reserve(raw.size());
    for (const RawDim &rd : raw)
      dims.push_back(make_input_dim(state, rd));
    ShapeTensor tshape = ShapeTensor::Tensor(dims);

    if (view.N != nullptr) {
      Dim Ndim = dims[0];
      DimEval Ne = eval_dim(Ndim, *state.symGraph);
      if (Ne.c.has_value() && *Ne.c != 1) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Input tensor (\"{}\") has fixed batch {} (unsupported).",
            name, *Ne.c));
      }
    }

    const unsigned C = get_const_channels(
        dims[view.C ? (raw.size() == 4 ? 1 : 0) : 0], name, state);

    Dim Hdim = dims[view.H ? (raw.size() == 4 ? 2 : 1) : 1];
    Dim Wdim = dims[view.W ? (raw.size() == 4 ? 3 : 2) : 2];

    Sym Hs = Hdim.isConst() ? Sym::Const(Hdim.value())
                            : state.symGraph->resolve(Hdim.sym());
    Sym Ws = Wdim.isConst() ? Sym::Const(Wdim.value())
                            : state.symGraph->resolve(Wdim.sym());

    std::optional<vkcnn::FloatType> hint = dtype_to_float_type(dtype);

    vkcnn::Tensor rt = state.output.input(C, std::nullopt, hint, Ws, Hs);
    auto t = Tensor::Runtime(rt, raw.size());
    state.tensors.map.emplace(name, std::move(t));
    return;
  }

  auto it = state.tensors.map.find(name);
  if (it == state.tensors.map.end()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output \"{}\" was never produced by any node.", name));
  }
  auto &produced = it->second;
  if (!produced.isRuntimeTensor()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Output (\"{}\") is not a runtime tensor (constant "
                    "outputs unsupported).",
                    name));
  }

  const ShapeTensor &outShape = produced.shape();
  if (!outShape.isTensor())
    throw std::runtime_error("vkcnn: Output shape must not be scalar.");
  const auto &outVec = outShape.dims();
  if (outVec.size() != 3 && outVec.size() != 4) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output tensor (\"{}\") must be rank 3 or 4; got {}.", name,
        outVec.size()));
  }

  unsigned int rtC = produced.runtime().tensor.channels();
  Symbolic rtH = produced.runtime().tensor.height();
  Symbolic rtW = produced.runtime().tensor.width();
  std::optional<vkcnn::FloatType> rtType = produced.runtime().tensor.type();

  if (view.C) {
    if (view.C->kind == RawDim::Kind::Value) {
      if (rtC != view.C->value) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Output (\"{}\") channel mismatch. Expected {}, got {}.",
            name, view.C->value, rtC));
      }
    } else if (view.C->kind == RawDim::Kind::Label) {
      auto itlab = state.symbolMap.find(view.C->label);
      if (itlab == state.symbolMap.end()) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Output (\"{}\") channel label \"{}\" not seen in inputs.",
            name, view.C->label));
      }
      auto re = state.symGraph->resolve(itlab->second);
      if (!re.isConstant() || re.constant() != rtC) {
        throw std::runtime_error(
            fmt::format("vkcnn: Output (\"{}\") channel mismatch (label {}).",
                        name, view.C->label));
      }
    }
  }

  check_output_axis(*view.H, rtH, state, "height", name);
  check_output_axis(*view.W, rtW, state, "width", name);

  if (view.N && view.N->kind == RawDim::Kind::Value && view.N->value != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Output (\"{}\") specifies batch {}, only 1 is supported.", name,
        view.N->value));
  }

  if (rtType.has_value()) {
    auto want = dtype_to_float_type(dtype);
    if (want.has_value() && want != rtType) {
      throw std::runtime_error("vkcnn: Output tensor type mismatch.");
    }
  } else {
    if (auto want = dtype_to_float_type(dtype)) {
      produced.runtime().tensor.setType(*want);
    }
  }
  state.output.output(produced.runtime().tensor);
}

} // namespace vkcnn::details
