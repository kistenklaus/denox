#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Cast(
    ImportState & /*state*/, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // Arity & output
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &inT = *inputs[0];

  // Device tensors unsupported here
  if (inT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": runtime tensors not supported.", node.name()));

  const HostTensor &in = inT.host();

  // Get 'to' attribute (required)
  auto it = attributes.find("to");
  if (it == attributes.end() || !it->second.isInt())
    throw std::runtime_error(
        fmt::format("vkcnn: Cast \"{}\": missing required int attribute 'to'.",
                    node.name()));

  // Map ONNX TensorProto_DataType (int) -> our Dtype
  const auto to_onxx = static_cast<int>(it->second.i());
  auto to_dtype_opt = parse_data_type(to_onxx);
  if (!to_dtype_opt.has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Cast \"{}\": unsupported target dtype {}.",
                    node.name(), to_onxx));
  const Dtype to_dtype = *to_dtype_opt;

  const Dtype src_dtype = in.type();

  // No-op if same dtype
  if (src_dtype == to_dtype)
    return {Tensor::Host(in)};

  // Dtype helpers
  auto is_integer_dtype = [](Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
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
  auto is_float_dtype = [](Dtype dt) {
    return dt == Dtype::Float16 || dt == Dtype::Float32 || dt == Dtype::Float64;
  };

  // Symbolic payload casting: follow your rules
  if (src_dtype == Dtype::Sym) {
    if (is_integer_dtype(to_dtype)) {
      // Treat as no-op (keep Sym representation)
      return {Tensor::Host(in)};
    }
    if (is_float_dtype(to_dtype) || to_dtype == Dtype::String) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Cast \"{}\": cannot cast Symbolic to float or string.",
          node.name()));
    }
    // Any other target kind is unsupported
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": unsupported cast from Sym.", node.name()));
  }

  // Strings: only string->string is a no-op (handled earlier); everything else
  // unsupported.
  if (src_dtype == Dtype::String) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": string casts are not supported.", node.name()));
  }

  // For now, only trivial no-op casts are supported beyond Sym rules.
  // (int<->int width changes, int<->float, float<->float width etc. are not
  // implemented)
  throw std::runtime_error(fmt::format(
      "vkcnn: Cast \"{}\": non-noop cast from {} to {} not supported yet.",
      node.name(), dtype_to_string(src_dtype), dtype_to_string(to_dtype)));
}

} // namespace vkcnn::details
