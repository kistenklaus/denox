#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
cast([[maybe_unused]] ImportState &state,
               memory::span<const memory::optional<Tensor>> inputs,
               std::size_t outputCount,
               const memory::hash_map<memory::string, Attribute> &attributes,
               [[maybe_unused]] opset_version version,
               memory::string_view nodeName) {

  // Arity & output
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Cast \"{}\" expects exactly 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &inT = *inputs[0];

  // Device tensors unsupported here
  if (inT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": runtime tensors not supported.", nodeName));

  const HostTensor &in = inT.host();

  // Get 'to' attribute (required)
  auto it = attributes.find("to");
  if (it == attributes.end() || !it->second.isInt())
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": missing required int attribute 'to'.", nodeName));

  // Map ONNX TensorProto_DataType (int) -> our Dtype
  const auto to_onxx = static_cast<int>(it->second.i());
  auto to_dtype_opt = Dtype::parse(to_onxx);
  if (!to_dtype_opt.has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": unsupported target dtype {}.", nodeName, to_onxx));
  const Dtype to_dtype = *to_dtype_opt;

  const Dtype src_dtype = in.type();

  // No-op if same dtype
  if (src_dtype == to_dtype)
    return {Tensor::Host(in)};

  // Symbolic payload casting: follow your rules
  if (src_dtype == Dtype::Sym) {
    if (to_dtype.isInteger()) {
      // Treat as no-op (keep Sym representation)
      return {Tensor::Host(in)};
    }
    if (to_dtype.isFloat() || to_dtype == Dtype::String) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Cast \"{}\": cannot cast Symbolic to float or string.",
          nodeName));
    }
    // Any other target kind is unsupported
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": unsupported cast from Sym.", nodeName));
  }

  // Strings: only string->string is a no-op (handled earlier); everything else
  // unsupported.
  if (src_dtype == Dtype::String) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Cast \"{}\": string casts are not supported.", nodeName));
  }

  // For now, only trivial no-op casts are supported beyond Sym rules.
  // (int<->int width changes, int<->float, float<->float width etc. are not
  // implemented)
  throw std::runtime_error(fmt::format(
      "vkcnn: Cast \"{}\": non-noop cast from {} to {} not supported yet.",
      nodeName, src_dtype.to_string(), to_dtype.to_string()));
}

} // namespace denox::onnx::details::ops
