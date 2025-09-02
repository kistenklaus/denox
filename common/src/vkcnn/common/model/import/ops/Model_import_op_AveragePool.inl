#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_AveragePool(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {
  // Arity / outputs: only the value output (no indices).
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: AveragePool \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: AveragePool \"{}\": expected exactly 1 output.", node.name()));

  const Tensor &X = *inputs[0];
  if (!X.isDevice())
    throw std::runtime_error(
        "vkcnn: AveragePool: only DeviceTensor is supported.");

  const DeviceTensor Xdev = X.device();
  const std::size_t r = Xdev.rank();
  if (r != 3 && r != 4)
    throw std::runtime_error(fmt::format(
        "vkcnn: AveragePool \"{}\": input must be CHW or NCHW.", node.name()));

  // --- Attributes ---

  // auto_pad: only NOTSET supported
  if (auto it = attributes.find("auto_pad"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": auto_pad must be string, got {}.",
          node.name(), AttributeKind_name(it->second.kind())));
    if (it->second.s() != "NOTSET")
      throw std::runtime_error(
          "vkcnn: AveragePool: only auto_pad=\"NOTSET\" is supported.");
  }

  // ceil_mode: must be 0/false
  if (auto it = attributes.find("ceil_mode"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": ceil_mode must be int, got {}.",
          node.name(), AttributeKind_name(it->second.kind())));
    if (it->second.i() != 0)
      throw std::runtime_error(
          "vkcnn: AveragePool: ceil_mode!=0 not supported.");
  }

  // count_include_pad: must be 0 (exclude pad) for now
  if (auto it = attributes.find("count_include_pad"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": count_include_pad must be int, got {}.",
          node.name(), AttributeKind_name(it->second.kind())));
    if (it->second.i() != 0)
      throw std::runtime_error(
          "vkcnn: AveragePool: count_include_pad=1 not supported.");
  }

  // Reject unexpected MaxPool-only attribute if present
  if (auto it = attributes.find("storage_order"); it != attributes.end()) {
    throw std::runtime_error(
        "vkcnn: AveragePool: storage_order is not a valid attribute.");
  }

  // Reject dilations if someone provided it (not in ONNX AveragePool)
  if (auto it = attributes.find("dilations"); it != attributes.end()) {
    throw std::runtime_error(
        "vkcnn: AveragePool: dilations attribute is not supported.");
  }

  // kernel_shape (required)
  glm::uvec2 kernel(0, 0);
  {
    auto it = attributes.find("kernel_shape");
    if (it == attributes.end())
      throw std::runtime_error("vkcnn: AveragePool: kernel_shape is required.");
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": kernel_shape must be ints, got {}.",
          node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(
          fmt::format("vkcnn: AveragePool \"{}\": kernel_shape must have size "
                      "2 (H,W), got {}.",
                      node.name(), v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: AveragePool: kernel dims must be >= 1.");
    kernel.y = static_cast<unsigned>(v[0]); // H
    kernel.x = static_cast<unsigned>(v[1]); // W
  }

  // strides (default 1,1)
  glm::uvec2 stride(1, 1);
  if (auto it = attributes.find("strides"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": strides must be ints, got {}.",
          node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": strides must have size 2 (H,W), got {}.",
          node.name(), v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: AveragePool: strides must be >= 1.");
    stride.y = static_cast<unsigned>(v[0]);
    stride.x = static_cast<unsigned>(v[1]);
  }

  // pads (default 0,0; allow symmetric 4-int form [top,left,bottom,right])
  glm::uvec2 padding(0, 0);
  if (auto it = attributes.find("pads"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: AveragePool \"{}\": pads must be ints, got {}.",
                      node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() == 2) {
      if (v[0] < 0 || v[1] < 0)
        throw std::runtime_error("vkcnn: AveragePool: pads must be >= 0.");
      padding.y = static_cast<unsigned>(v[0]); // H
      padding.x = static_cast<unsigned>(v[1]); // W
    } else if (v.size() == 4) {
      if (v[0] != v[2] || v[1] != v[3])
        throw std::runtime_error(
            "vkcnn: AveragePool: asymmetric pads not supported "
            "(require top==bottom and left==right).");
      if (v[0] < 0 || v[1] < 0)
        throw std::runtime_error("vkcnn: AveragePool: pads must be >= 0.");
      padding.y = static_cast<unsigned>(v[0]); // top/bottom
      padding.x = static_cast<unsigned>(v[1]); // left/right
    } else {
      throw std::runtime_error(fmt::format(
          "vkcnn: AveragePool \"{}\": pads must have size 2 or 4, got {}.",
          node.name(), v.size()));
    }
  }

  // Backend call — fix dilation to (1,1) for AveragePool
  const glm::uvec2 dilation(1, 1);
  vkcnn::Tensor outHandle = state.output.pool(
      Xdev.handle(), kernel, padding, stride, dilation, PoolFunction::Avg);

  return {Tensor::Device(DeviceTensor{Xdev.rank(), std::move(outHandle)})};
}

} // namespace vkcnn::details
