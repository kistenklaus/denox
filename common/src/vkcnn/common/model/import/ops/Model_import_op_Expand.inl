#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Expand(
    [[maybe_unused]] ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Attribute> & attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {
  // ---- arity ----
  if (inputs.size() != 2)
    throw std::runtime_error(fmt::format(
        "vkcnn: Expand \"{}\" expects 2 inputs (data, shape).", node.name()));
  if (!inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Expand \"{}\": both data and shape are required.",
                    node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Expand \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];
  const Tensor &shapeT = *inputs[1];

  // Only HostTensor data supported for now
  if (!dataT.isHost())
    throw std::runtime_error(
        fmt::format("vkcnn: Expand \"{}\": DeviceTensor data is not supported.",
                    node.name()));
  if (!shapeT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Expand \"{}\": shape must be a HostTensor.", node.name()));

  const HostTensor &Xin = dataT.host();
  const HostTensor &shapeH = shapeT.host();

  // ---- helpers to read 1-D INT64 control tensor (respecting view/offset) ----
  auto read_len_1d = [](const HostTensor &t) -> std::size_t {
    if (!t.isConstant())
      throw std::runtime_error(
          "vkcnn: Expand: control tensor must have constant shape.");
    const auto dims = t.shape().toU64();
    if (dims.size() != 1)
      throw std::runtime_error("vkcnn: Expand: 'shape' must be 1-D.");
    return static_cast<std::size_t>(dims[0]);
  };

  auto read_i64_1d = [&](const HostTensor &t) -> std::vector<std::int64_t> {
    if (t.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Expand \"{}\": 'shape' must be INT64.", node.name()));
    const std::size_t n = read_len_1d(t);
    std::vector<std::int64_t> out(n);
    if (t.isContiguous() && t.view().offset().isConstant() &&
        t.view().offset().constant() == 0) {
      auto s = t.storage()->i64();
      if (s.size() < n)
        throw std::runtime_error(
            "vkcnn: Expand: storage smaller than logical size.");
      std::memcpy(out.data(), s.data(), n * sizeof(std::int64_t));
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t ui = static_cast<std::uint64_t>(i);
        const std::size_t elem = t.view().constIndexOf({&ui, 1});
        out[i] = t.storage()->i64()[elem];
      }
    }
    return out;
  };

  // ---- parse target shape ----
  auto shapeVals = read_i64_1d(shapeH);
  for (std::size_t i = 0; i < shapeVals.size(); ++i) {
    if (shapeVals[i] < 0)
      throw std::runtime_error(
          fmt::format("vkcnn: Expand \"{}\": 'shape'[{}] is negative ({}).",
                      node.name(), i, shapeVals[i]));
  }

  auto g = Xin.shape().graph();
  TensorShape outShape(
      g, std::span<const std::int64_t>(shapeVals.data(), shapeVals.size()));

  // ---- broadcasting validation (NumPy-style, right-aligned) ----
  const TensorShape inShape = Xin.shape();
  const std::size_t rIn = inShape.rank();
  const std::size_t rOut = outShape.rank();

  if (rOut < rIn) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Expand \"{}\": target rank ({}) must be >= input rank ({}).",
        node.name(), rOut, rIn));
  }

  // Map each input axis i -> output axis j = (rOut - rIn) + i
  std::vector<int64_t> axesMap;
  axesMap.reserve(rIn);
  for (std::size_t i = 0; i < rIn; ++i) {
    axesMap.push_back(static_cast<int64_t>((rOut - rIn) + i));
  }

  // Validate each mapped pair (inDim, outDim): allow if in==1 or in==out
  for (std::size_t i = 0; i < rIn; ++i) {
    const auto &inDim = inShape[i];
    const auto &outDim = outShape[static_cast<std::size_t>(axesMap[i])];

    const bool inIsOne = (inDim.isConstant() && inDim.constant() == 1);
    if (inIsOne)
      continue;

    if (inDim == outDim)
      continue;

    // Otherwise not broadcast-compatible
    throw std::runtime_error(fmt::format(
        "vkcnn: Expand \"{}\": input dim {} is incompatible with target dim {} "
        "at axis {}.",
        node.name(),
        (inDim.isConstant() ? std::to_string(inDim.constant())
                            : std::string("sym")),
        (outDim.isConstant() ? std::to_string(outDim.constant())
                             : std::string("sym")),
        i));
  }

  // ---- build broadcasted view ----
  const TensorViewDesc inView = Xin.view();
  const TensorViewDesc outView =
      inView.broadcastInDim(inShape.dims(), outShape.dims(), axesMap);

  // ---- produce host tensor with new shape/view (no materialization) ----
  HostTensor out = Xin.withView(std::move(outShape), std::move(outView));
  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
