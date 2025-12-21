#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"
#include "denox/common/ActivationFunction.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> leaky_relu(
    ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // Arity
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: LeakyRelu \"{}\" expects exactly 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: LeakyRelu \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &inT = *inputs[0];
  if (!inT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: LeakyRelu \"{}\": only runtime tensors are supported.",
        nodeName));

  [[maybe_unused]] float alpha = 0.01f;
  if (auto it = attributes.find("alpha"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isFloat())
      throw std::runtime_error(fmt::format(
          "vkcnn: LeakyRelu \"{}\": attribute 'alpha' must be float.",
          nodeName));
    alpha = a.f();
  }

  const DeviceTensor &inDev = inT.device();
  const std::size_t r = inDev.rank();
  // TODO: Remodel activation function to include parameters.
  compiler::TensorHandle outHandle = state.output.activation(
      inDev.handle(), ActivationFunction::LeakyReLU);

  DeviceTensor outDev(r, std::move(outHandle));
  return {Tensor::Device(std::move(outDev))};
}

} // namespace denox::onnx::details::ops
