#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Transpose(
    [[maybe_unused]] ImportState & /*state*/,
    std::span<const std::optional<Tensor>> inputs, std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // Arity checks
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Transpose \"{}\" expects exactly 1 input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Transpose \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &inT = *inputs[0];
  if (inT.isDevice())
    throw std::runtime_error(
        fmt::format("vkcnn: Transpose \"{}\": runtime tensors not supported.",
                    node.name()));
  const HostTensor &X = inT.host();

  const std::size_t r = X.rank();

  // Read perm (optional). Default: reverse axes.
  std::vector<std::int64_t> perm;
  if (auto it = attributes.find("perm"); it != attributes.end()) {
    if (!it->second.isInts())
      throw std::runtime_error(fmt::format(
          "vkcnn: Transpose \"{}\": 'perm' must be a list of int64.",
          node.name()));
    const auto &v = it->second.ints();
    if (v.empty()) {
      // Treat empty provided perm as "use default" (reverse)
      perm.resize(r);
      for (std::size_t i = 0; i < r; ++i)
        perm[i] = static_cast<std::int64_t>(r - 1 - i);
    } else {
      perm.assign(v.begin(), v.end());
    }
  } else {
    perm.resize(r);
    for (std::size_t i = 0; i < r; ++i)
      perm[i] = static_cast<std::int64_t>(r - 1 - i);
  }

  // Validate perm is a permutation of [0..r-1]
  if (perm.size() != r)
    throw std::runtime_error(fmt::format(
        "vkcnn: Transpose \"{}\": perm size ({}) must equal rank ({}).",
        node.name(), perm.size(), r));
  {
    std::vector<uint8_t> seen(r, 0);
    for (auto v : perm) {
      if (v < 0 || static_cast<std::size_t>(v) >= r)
        throw std::runtime_error(fmt::format(
            "vkcnn: Transpose \"{}\": perm value {} out of range for rank {}.",
            node.name(), v, r));
      if (seen[static_cast<std::size_t>(v)]++)
        throw std::runtime_error(fmt::format(
            "vkcnn: Transpose \"{}\": perm contains duplicates.", node.name()));
    }
  }

  // Zero-copy: permute shape and view
  auto newShape = X.shape().permute(perm);
  auto newView = X.view().permute(perm);
  HostTensor out = X.withView(std::move(newShape), std::move(newView));

  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
