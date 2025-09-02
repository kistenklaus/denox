#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Relu(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  // Arity
  if (inputs.size() != 1 || !inputs[0].has_value()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\" expects exactly 1 input.", node.name()));
  }
  if (outputCount != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\" must have exactly 1 output.", node.name()));
  }

  // No attributes for ONNX Relu
  if (!attributes.empty()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Relu \"{}\": unexpected attributes present.", node.name()));
  }

  const Tensor &inT = *inputs[0];
  if (!inT.isDevice()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Relu \"{}\": only runtime tensors are supported.",
                    node.name()));
  }

  const DeviceTensor &inDev = inT.device();
  const std::size_t r = inDev.rank();

  vkcnn::Tensor outHandle =
      state.output.activation(inDev.handle(), ActivationFunction::ReLU);

  DeviceTensor outDev(r, std::move(outHandle));
  return {Tensor::Device(std::move(outDev))};
}

} // namespace vkcnn::details
