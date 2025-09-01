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

static std::vector<Tensor>
import_op_Constant([[maybe_unused]] ImportState &state,
                   std::span<const std::optional<Tensor>> inputs,
                   std::size_t outputCount,
                   const std::unordered_map<std::string, Tensor> &attributes,
                   opset_version version, const onnx::NodeProto & /*node*/) {

  // --- Contract checks
  if (!inputs.empty())
    throw std::runtime_error("vkcnn: Constant takes 0 inputs");
  if (outputCount != 1)
    throw std::runtime_error("vkcnn: Constant must produce exactly 1 output");

  auto has = [&](const char *k) {
    return attributes.find(k) != attributes.end();
  };
  const bool h_value = has("value");
  const bool h_sparse_value = has("sparse_value");
  const bool h_vint = has("value_int");
  const bool h_vints = has("value_ints");
  const bool h_vfloat = has("value_float");
  const bool h_vfloats = has("value_floats");
  const bool h_vstring = has("value_string");
  const bool h_vstrings = has("value_strings");

  // (Non-standard extension) allow a list of TensorProto via attribute decoding
  const bool h_vtensors = has("value_tensors");

  const int count_star = int(h_vint) + int(h_vints) + int(h_vfloat) +
                         int(h_vfloats) + int(h_vstring) + int(h_vstrings);

  auto bad_combo = [&](int cnt) {
    if (cnt != 1)
      throw std::runtime_error(
          "vkcnn: Constant must specify exactly one of "
          "'value', 'sparse_value', or one 'value_*' attribute");
  };

  if (version < 11) {
    if (!h_value || h_sparse_value || count_star)
      throw std::runtime_error(
          "vkcnn: Constant(opset<11) requires only 'value'");
  } else if (version < 12) {
    bad_combo(int(h_value) + int(h_sparse_value));
    if (count_star)
      throw std::runtime_error(
          "vkcnn: Constant(opset=11) does not support 'value_*'");
  } else {
    bad_combo(int(h_value) + int(h_sparse_value) + (count_star ? 1 : 0));
  }

  if (h_sparse_value)
    throw std::runtime_error(
        "vkcnn: Constant with 'sparse_value' not supported");

  // --- Helpers
  auto to_bytes = [](auto const &vec) {
    std::vector<std::byte> out(
        sizeof(typename std::decay_t<decltype(vec)>::value_type) * vec.size());
    std::memcpy(out.data(), vec.data(), out.size());
    return out;
  };

  auto make_vec_shape = [](std::size_t n) {
    ShapeVector dims;
    dims.reserve(1);
    dims.push_back(Dim::Const(static_cast<uint64_t>(n)));
    return ShapeTensor::Tensor(std::move(dims)); // 1-D [n]
  };

  // --- value: TensorProto → RawTensor (any rank/any dtype)
  if (h_value) {
    const Tensor &t = attributes.at("value");
    if (t.isUnknown())
      throw std::runtime_error("vkcnn: Constant 'value' is unknown");

    // If your attribute decoder already turned TensorProto into RawTensor, just
    // pass it through.
    if (t.isRaw()) {
      return {t}; // already a RawTensor
    }

    // Tolerate 0-D scalar encodings (some exporters encode 0-D via scalar attr
    // decoding)
    if (t.isScalar()) {
      const auto &s = t.scalar();
      switch (s.dtype) {
      case Dtype::Int64:
        return {Tensor::Scalar(s.v.i)};
      case Dtype::Int32:
      case Dtype::Int16:
        return {Tensor::Scalar(static_cast<std::int64_t>(s.v.i))};
      case Dtype::Float32:
        return {Tensor::Scalar(s.v.float32)};
      case Dtype::Float64:
        return {Tensor::Scalar(static_cast<f32>(s.v.float64))};
      default:
        break;
      }
      throw std::runtime_error(
          "vkcnn: Constant 'value' scalar: unsupported dtype");
    }
    if (t.isString()) {
      // Treat as 0-D tensor(string)
      return {t};
    }
    if (t.isList()) {
      // Rare/extension: allow a list to produce a ListTensor constant
      return {t};
    }
    throw std::runtime_error(
        "vkcnn: Constant 'value' has unsupported internal kind");
  }

  // --- value_int → scalar int64
  if (h_vint) {
    const Tensor &t = attributes.at("value_int");
    if (t.isUnknown() || !t.isScalar())
      throw std::runtime_error(
          "vkcnn: Constant 'value_int' must be an integer scalar");
    const auto &s = t.scalar();
    if (s.dtype != Dtype::Int64 && s.dtype != Dtype::Int32 &&
        s.dtype != Dtype::Int16)
      throw std::runtime_error(
          "vkcnn: Constant 'value_int' must be an integer");
    return {Tensor::Scalar(static_cast<std::int64_t>(s.v.i))};
  }

  // --- value_float → scalar float32 (allow f64 downcast)
  if (h_vfloat) {
    const Tensor &t = attributes.at("value_float");
    if (t.isUnknown() || !t.isScalar())
      throw std::runtime_error(
          "vkcnn: Constant 'value_float' must be a float scalar");
    const auto &s = t.scalar();
    if (s.dtype == Dtype::Float32)
      return {Tensor::Scalar(s.v.float32)};
    if (s.dtype == Dtype::Float64)
      return {Tensor::Scalar(static_cast<f32>(s.v.float64))};
    throw std::runtime_error(
        "vkcnn: Constant 'value_float' must be float32/float64");
  }

  // --- value_string → scalar string
  if (h_vstring) {
    const Tensor &t = attributes.at("value_string");
    if (t.isUnknown() || !t.isString())
      throw std::runtime_error(
          "vkcnn: Constant 'value_string' must be a string scalar");
    return {Tensor::String(t.string().str)};
  }

  // --- value_ints → 1-D RawTensor(int64)
  if (h_vints) {
    const Tensor &t = attributes.at("value_ints");
    if (!t.isList())
      throw std::runtime_error("vkcnn: Constant 'value_ints' must be a list");
    std::vector<std::int64_t> vals;
    for (const Tensor &e : t.list().tensors) {
      if (e.isUnknown() || !e.isScalar())
        throw std::runtime_error(
            "vkcnn: Constant 'value_ints' elements must be integer scalars");
      const auto &s = e.scalar();
      if (s.dtype != Dtype::Int64 && s.dtype != Dtype::Int32 &&
          s.dtype != Dtype::Int16)
        throw std::runtime_error(
            "vkcnn: Constant 'value_ints' elements must be integer");
      vals.push_back(static_cast<std::int64_t>(s.v.i));
    }
    RawTensor rt{.shape = make_vec_shape(vals.size()),
                 .type = Dtype::Int64,
                 .raw = to_bytes(vals)};
    return {Tensor::Raw(std::move(rt))};
  }

  // --- value_floats → 1-D RawTensor(float32) (downcast f64)
  if (h_vfloats) {
    const Tensor &t = attributes.at("value_floats");
    if (!t.isList())
      throw std::runtime_error("vkcnn: Constant 'value_floats' must be a list");
    std::vector<f32> vals;
    for (const Tensor &e : t.list().tensors) {
      if (e.isUnknown() || !e.isScalar())
        throw std::runtime_error(
            "vkcnn: Constant 'value_floats' elements must be float scalars");
      const auto &s = e.scalar();
      if (s.dtype == Dtype::Float32)
        vals.push_back(s.v.float32);
      else if (s.dtype == Dtype::Float64)
        vals.push_back(static_cast<f32>(s.v.float64));
      else
        throw std::runtime_error(
            "vkcnn: Constant 'value_floats' elements must be float32/float64");
    }
    RawTensor rt{.shape = make_vec_shape(vals.size()),
                 .type = Dtype::Float32,
                 .raw = to_bytes(vals)};
    return {Tensor::Raw(std::move(rt))};
  }

  // --- value_strings → (temporary) ListTensor of scalar strings
  // If you later add an encoding for string RawTensors, convert to 1-D
  // RawTensor{type=String}.
  if (h_vstrings) {
    const Tensor &t = attributes.at("value_strings");
    if (!t.isList())
      throw std::runtime_error(
          "vkcnn: Constant 'value_strings' must be a list");
    for (const Tensor &e : t.list().tensors) {
      if (e.isUnknown() || !e.isString())
        throw std::runtime_error(
            "vkcnn: Constant 'value_strings' elements must be strings");
    }
    return {t}; // ListTensor of scalar strings
  }

  // --- (Extension) value_tensors → sequence(ListTensor)
  if (h_vtensors) {
    const Tensor &t = attributes.at("value_tensors");
    if (!t.isList())
      throw std::runtime_error(
          "vkcnn: Constant 'value_tensors' must be a list");
    return {t}; // ListTensor (sequence)
  }

  // Should be unreachable (opset gating guarantees one branch above)
  throw std::runtime_error("vkcnn: Constant: missing supported attribute");
}

} // namespace vkcnn::details
