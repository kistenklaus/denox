#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_AffineGrid(
    [[maybe_unused]] ImportState &state,
    [[maybe_unused]] std::span<const std::optional<Tensor>> inputs,
    [[maybe_unused]] std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {
  throw std::runtime_error(fmt::format(
      "vkcnn: operation AffineGrid is not supported (node = \"{}\")", node.name()));
}

} // namespace vkcnn::details
