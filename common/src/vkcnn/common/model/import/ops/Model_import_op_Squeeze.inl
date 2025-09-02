#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Squeeze(
    [[maybe_unused]] ImportState &state,
    std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Attribute> & attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  // ---- arity ----
  if (inputs.size() < 1 || inputs.size() > 2)
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\" expects 1 or 2 inputs (data, [axes]).", node.name()));
  if (!inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\": data is required.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];

  // Only HostTensor supported per requirements.
  if (dataT.isDevice()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\": DeviceTensor not supported.", node.name()));
  }
  if (!dataT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\": expected HostTensor.", node.name()));

  const HostTensor &Xin = dataT.host();
  const TensorShape inShape = Xin.shape();
  const std::size_t r = inShape.rank();

  // ---- helpers to read 1-D INT64 with view/offset respected ----
  auto read_len_1d = [](const HostTensor &t) -> std::size_t {
    if (!t.isConstant())
      throw std::runtime_error(
          "vkcnn: Squeeze: control tensor must have constant shape.");
    const auto dims = t.shape().toU64();
    if (dims.size() != 1)
      throw std::runtime_error("vkcnn: Squeeze: 'axes' must be 1-D.");
    return static_cast<std::size_t>(dims[0]);
  };

  auto read_i64_1d = [&](const HostTensor &t) -> std::vector<std::int64_t> {
    if (t.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Squeeze \"{}\": 'axes' must be INT64.", node.name()));
    const std::size_t n = read_len_1d(t);
    std::vector<std::int64_t> out(n);
    if (t.isContiguous() && t.view().offset().isConstant() &&
        t.view().offset().constant() == 0) {
      auto s = t.storage()->i64();
      if (s.size() < n)
        throw std::runtime_error(
            "vkcnn: Squeeze: storage smaller than logical size.");
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

  // ---- read axes from input (if provided) ----
  std::vector<std::size_t> axes; // normalized, unique, sorted descending
  if (inputs.size() == 2 && inputs[1].has_value()) {
    const Tensor &axesT = *inputs[1];
    if (!axesT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Squeeze \"{}\": 'axes' input must be a HostTensor.", node.name()));
    const auto axesI64 = read_i64_1d(axesT.host());

    // normalize and validate
    std::vector<std::size_t> tmp;
    tmp.reserve(axesI64.size());
    for (auto a : axesI64) {
      if (a < 0) a += static_cast<std::int64_t>(r);
      if (a < 0 || a >= static_cast<std::int64_t>(r))
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": axis {} out of range for rank {}.",
            node.name(), a, r));
      tmp.push_back(static_cast<std::size_t>(a));
    }
    // duplicates?
    std::vector<uint8_t> seen(r, 0);
    for (auto a : tmp) {
      if (seen[a]++)
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": duplicate axis {}.", node.name(), a));
    }
    // ONNX: each specified axis must be of size 1 (constant)
    for (auto a : tmp) {
      const auto &d = inShape[a];
      if (!d.isConstant())
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": axis {} is symbolic; must be constant 1.",
            node.name(), a));
      if (d.constant() != 1)
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": dimension at axis {} is {}, expected 1.",
            node.name(), a, d.constant()));
    }
    axes = std::move(tmp);
  } else {
    // no axes provided → remove every const-1 dimension
    for (std::size_t i = 0; i < r; ++i) {
      const auto &d = inShape[i];
      if (d.isConstant() && d.constant() == 1)
        axes.push_back(i);
    }
  }

  // If nothing to squeeze, return identity
  if (axes.empty()) {
    return {Tensor::Host(Xin)};
  }

  // Sort descending so we can erase safely
  std::sort(axes.begin(), axes.end(), std::greater<std::size_t>());

  // ---- build new shape/view ----
  TensorShape outShape = inShape;
  TensorViewDesc outView = Xin.view();
  for (std::size_t ax : axes) {
    outShape = outShape.squeeze(ax);
    outView  = outView.squeeze(ax);
  }

  HostTensor out = Xin.withView(std::move(outShape), std::move(outView));
  return {Tensor::Host(std::move(out))};
}
} // namespace vkcnn::details
