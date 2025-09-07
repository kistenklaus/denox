#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
relu(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // Arity
  if (inputs.size() != 1 || !inputs[0].has_value()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Relu \"{}\" expects exactly 1 input.", nodeName));
  }
  if (outputCount != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\" must have exactly 1 output.", nodeName));
  }

  // No attributes for ONNX Relu
  if (!attributes.empty()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\": unexpected attributes present.", nodeName));
  }

  const Tensor &inT = *inputs[0];
  if (!inT.isDevice()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\": only runtime tensors are supported.", nodeName));
  }

  const DeviceTensor &inDev = inT.device();
  const std::size_t r = inDev.rank();

  compiler::Tensor outHandle = state.output.activation(
      inDev.handle(), compiler::ActivationFunction::ReLU);

  DeviceTensor outDev(r, std::move(outHandle));
  return {Tensor::Device(std::move(outDev))};
}

} // namespace denox::onnx::details::ops
