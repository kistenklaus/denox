#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Slice(
    ImportState & /*state*/, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // ---- arity & kinds ----
  if (inputs.size() < 1 || !inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Slice \"{}\" expects data input.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];
  if (dataT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": runtime tensors not supported.", node.name()));
  const HostTensor &data = dataT.host();

  // Need static data shape (we compute sizes/clamps)
  if (!data.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": data must have static shape.", node.name()));

  // ---- fetch starts/ends/axes/steps ----
  std::vector<std::int64_t> starts, ends, axes, steps;

  auto read_vec_from_tensor_i64 = [&](const Tensor &t, const char *what) {
    if (t.isDevice()) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": {} must be host INT64.", node.name(), what));
    }
    const HostTensor &h = t.host();
    if (!h.isConstant() || h.type() != Dtype::Int64)
      throw std::runtime_error(
          fmt::format("vkcnn: Slice \"{}\": {} must be constant INT64.",
                      node.name(), what));
    auto sp = h.storage()->i64();
    return std::vector<std::int64_t>(sp.begin(), sp.end());
  };

  const bool have_inputs_style =
      (inputs.size() >= 3 && inputs[1].has_value() && inputs[2].has_value());

  if (have_inputs_style) {
    // opset >= 10 style: data, starts, ends, [axes], [steps]
    starts = read_vec_from_tensor_i64(*inputs[1], "starts");
    ends = read_vec_from_tensor_i64(*inputs[2], "ends");
    if (inputs.size() >= 4 && inputs[3].has_value())
      axes = read_vec_from_tensor_i64(*inputs[3], "axes");
    if (inputs.size() >= 5 && inputs[4].has_value())
      steps = read_vec_from_tensor_i64(*inputs[4], "steps");
  } else {
    // legacy attributes style
    auto pick_ints = [&](const char *key) -> std::vector<std::int64_t> {
      auto it = attributes.find(key);
      if (it == attributes.end())
        return {};
      if (!it->second.isInts())
        throw std::runtime_error(
            fmt::format("vkcnn: Slice \"{}\": attribute '{}' must be ints.",
                        node.name(), key));
      return it->second.ints();
    };
    starts = pick_ints("starts");
    ends = pick_ints("ends");
    axes = pick_ints("axes");
    steps = pick_ints("steps");
  }

  if (starts.empty() || ends.empty() || starts.size() != ends.size())
    throw std::runtime_error(
        fmt::format("vkcnn: Slice \"{}\": starts/ends must be provided and "
                    "have same length.",
                    node.name()));

  // defaults
  const auto inDims = data.shape().toU64();
  const std::size_t rank = inDims.size();

  if (axes.empty()) {
    axes.resize(starts.size());
    for (std::size_t i = 0; i < axes.size(); ++i)
      axes[i] = static_cast<std::int64_t>(i);
  }
  if (steps.empty())
    steps.assign(starts.size(), 1);

  if (!(axes.size() == starts.size() && steps.size() == starts.size()))
    throw std::runtime_error(
        fmt::format("vkcnn: Slice \"{}\": axes/steps (if present) must match "
                    "starts/ends length.",
                    node.name()));

  // normalize axes (allow negative), check dupes
  {
    std::vector<char> seen(rank, 0);
    for (auto &ax : axes) {
      if (ax < 0)
        ax += static_cast<std::int64_t>(rank);
      if (ax < 0 || static_cast<std::size_t>(ax) >= rank)
        throw std::runtime_error(fmt::format(
            "vkcnn: Slice \"{}\": axis {} out of range.", node.name(), ax));
      if (seen[(std::size_t)ax]++)
        throw std::runtime_error(fmt::format(
            "vkcnn: Slice \"{}\": duplicate axis {}.", node.name(), ax));
    }
  }

  // ---- compute new view & shape ----
  auto g = data.shape().graph();
  TensorShape curShape = data.shape();
  TensorViewDesc curView = data.view();

  // keep a mutable copy of current dims for per-axis D
  auto curDims = curShape.toU64();
  bool usedNegativeStep = false;

  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t axis = static_cast<std::size_t>(axes[k]);
    const std::uint64_t D = curDims[axis];

    std::int64_t st = steps[k];
    if (st == 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": step=0 on axis {}.", node.name(), axis));

    std::int64_t s = starts[k];
    std::int64_t e = ends[k];

    // ---- normalize indices per ONNX ----
    if (st > 0) {
      // positive step: clamp s,e to [0, D]
      if (s < 0)
        s += static_cast<std::int64_t>(D);
      if (e < 0)
        e += static_cast<std::int64_t>(D);
      if (s < 0)
        s = 0;
      if (s > static_cast<std::int64_t>(D))
        s = static_cast<std::int64_t>(D);
      if (e < 0)
        e = 0;
      if (e > static_cast<std::int64_t>(D))
        e = static_cast<std::int64_t>(D);

      std::uint64_t L = 0;
      if (e > s) {
        const std::int64_t span = e - s;
        const std::int64_t step = st;
        L = static_cast<std::uint64_t>((span + step - 1) / step);
      }

      // apply view slice
      curView = curView.slice(axis, Symbolic{g, Sym::Const(s)},
                              Symbolic{g, Sym::Const(st)});
      // update shape
      curShape[axis] = Symbolic{g, Sym::Const(static_cast<std::int64_t>(L))};
      curDims[axis] = L;
    } else {
      // negative step: normalize for descending
      usedNegativeStep = true;
      const std::int64_t stepAbs = -st;

      if (s < 0)
        s += static_cast<std::int64_t>(D);
      if (e < 0)
        e += static_cast<std::int64_t>(D);

      if (D == 0) {
        // Empty dim → length stays 0; any view is fine
        curShape[axis] = Symbolic{g, Sym::Const(0)};
        curDims[axis] = 0;
        continue;
      }

      // clamp s to [0, D-1], e to [-1, D-1]
      if (s < 0)
        s = 0;
      if (s > static_cast<std::int64_t>(D) - 1)
        s = static_cast<std::int64_t>(D) - 1;
      if (e < -1)
        e = -1;
      if (e > static_cast<std::int64_t>(D) - 1)
        e = static_cast<std::int64_t>(D) - 1;

      std::uint64_t L = 0;
      if (s > e) {
        const std::int64_t span = (s - e);
        L = static_cast<std::uint64_t>((span + stepAbs - 1) / stepAbs);
      }

      // Implement via reverse + forward slice:
      // reverse -> start' = (D-1 - s), step' = +|st|
      const std::int64_t startFwd = static_cast<std::int64_t>(D - 1) - s;
      // (end is captured by L; view doesn't need it)

      curView = curView.reverse(
          axis, Symbolic{g, Sym::Const(static_cast<std::int64_t>(D))});
      curView = curView.slice(axis, Symbolic{g, Sym::Const(startFwd)},
                              Symbolic{g, Sym::Const(stepAbs)});

      curShape[axis] = Symbolic{g, Sym::Const(static_cast<std::int64_t>(L))};
      curDims[axis] = L;
    }
  }

  // keep strides positive if we used any negative step
  if (usedNegativeStep && curView.hasNegativeStride()) {
    curView = curView.normalizeNegativeStrides(curShape.dims());
  }

  // return a view on the same storage
  HostTensor out = data.withView(curShape, curView);
  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
