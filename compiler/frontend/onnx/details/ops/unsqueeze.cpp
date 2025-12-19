#include "denox/algorithm/unstable_sort.hpp"
#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> unsqueeze(
    [[maybe_unused]] ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // Arity & output count
  if (inputs.empty() || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Unsqueeze \"{}\" expects at least 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Unsqueeze \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &dataT = *inputs[0];

  // Device tensors not supported here
  if (dataT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Unsqueeze \"{}\": runtime tensors not supported.", nodeName));

  const HostTensor &data = dataT.host();

  // ---- Get axes (prefer input[1] if present; else attribute "axes") ----
  memory::vector<int64_t> axes;

  if (inputs.size() >= 2 && inputs[1].has_value()) {
    const Tensor &axesT = *inputs[1];
    if (axesT.isDevice())
      throw std::runtime_error(fmt::format(
          "vkcnn: Unsqueeze \"{}\": axes input must be host tensor.",
          nodeName));
    const HostTensor &axesH = axesT.host();
    if (!axesH.isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: Unsqueeze \"{}\": axes must be a constant tensor.",
          nodeName));
    if (axesH.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Unsqueeze \"{}\": axes tensor must be INT64.", nodeName));

    HostTensor axesC = axesH.contiguous();
    const auto span = axesC.storage()->i64();
    axes.assign(span.begin(), span.end());
  } else {
    auto it = attributes.find("axes");
    if (it == attributes.end() || !it->second.isInts())
      throw std::runtime_error(fmt::format(
          "vkcnn: Unsqueeze \"{}\": missing 'axes' (as input or attribute).",
          nodeName));
    const auto &v = it->second.ints();
    axes.assign(v.begin(), v.end());
  }

  // Empty axes: no-op
  if (axes.empty())
    return {Tensor::Host(data)};

  // ---- Normalize & validate axes ----
  const int64_t inRank = static_cast<int64_t>(data.rank());

  // Allowed range per ONNX: [-inRank-1, inRank]
  for (auto &ax : axes) {
    if (ax < -(inRank + 1) || ax > inRank)
      throw std::runtime_error(
          fmt::format("vkcnn: Unsqueeze \"{}\": axis {} out of valid range "
                      "[-{}, {}] for rank {}.",
                      nodeName, ax, inRank + 1, inRank, inRank));
    if (ax < 0)
      ax += (inRank + 1); // normalize to [0..inRank]
  }

  // Axes must be unique
  {
    memory::vector<int64_t> tmp = axes;
    algorithm::unstable_sort(tmp.begin(), tmp.end());
    if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
      throw std::runtime_error(fmt::format(
          "vkcnn: Unsqueeze \"{}\": axes must be unique.", nodeName));
  }

  // ---- Apply unsqueeze (view-only; no copy) ----
  // Insert in ascending order; compensate for prior insertions with an offset.
  std::sort(axes.begin(), axes.end());
  HostTensor out = data;
  int64_t inserted = 0;
  for (int64_t ax : axes) {
    const size_t where = static_cast<size_t>(ax + inserted);
    out = out.unsqueeze(where);
    ++inserted;
  }

  return {Tensor::Host(std::move(out))};
}

} // namespace denox::onnx::details::ops
