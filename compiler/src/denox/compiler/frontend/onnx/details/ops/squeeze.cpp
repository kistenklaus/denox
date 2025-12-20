#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"
#include "denox/algorithm/unstable_sort.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
squeeze([[maybe_unused]] ImportState &state,
        memory::span<const memory::optional<Tensor>> inputs,
        std::size_t outputCount,
        [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
            &attributes,
        [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // ---- arity ----
  if (inputs.size() < 1 || inputs.size() > 2)
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\" expects 1 or 2 inputs (data, [axes]).",
        nodeName));
  if (!inputs[0].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Squeeze \"{}\": data is required.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &dataT = *inputs[0];

  // Only HostTensor supported per requirements.
  if (dataT.isDevice()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Squeeze \"{}\": DeviceTensor not supported.", nodeName));
  }
  if (!dataT.isHost())
    throw std::runtime_error(
        fmt::format("vkcnn: Squeeze \"{}\": expected HostTensor.", nodeName));

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

  auto read_i64_1d = [&](const HostTensor &t) -> memory::vector<std::int64_t> {
    if (t.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Squeeze \"{}\": 'axes' must be INT64.", nodeName));
    const std::size_t n = read_len_1d(t);
    memory::vector<std::int64_t> out(n);
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
  memory::vector<std::size_t> axes; // normalized, unique, sorted descending
  if (inputs.size() == 2 && inputs[1].has_value()) {
    const Tensor &axesT = *inputs[1];
    if (!axesT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Squeeze \"{}\": 'axes' input must be a HostTensor.",
          nodeName));
    const auto axesI64 = read_i64_1d(axesT.host());

    // normalize and validate
    memory::vector<std::size_t> tmp;
    tmp.reserve(axesI64.size());
    for (auto a : axesI64) {
      if (a < 0)
        a += static_cast<std::int64_t>(r);
      if (a < 0 || a >= static_cast<std::int64_t>(r))
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": axis {} out of range for rank {}.",
            nodeName, a, r));
      tmp.push_back(static_cast<std::size_t>(a));
    }
    // duplicates?
    memory::vector<uint8_t> seen(r, 0);
    for (auto a : tmp) {
      if (seen[a]++)
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": duplicate axis {}.", nodeName, a));
    }
    // ONNX: each specified axis must be of size 1 (constant)
    for (auto a : tmp) {
      const auto &d = inShape[a];
      if (!d.isConstant())
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": axis {} is symbolic; must be constant 1.",
            nodeName, a));
      if (d.constant() != 1)
        throw std::runtime_error(fmt::format(
            "vkcnn: Squeeze \"{}\": dimension at axis {} is {}, expected 1.",
            nodeName, a, d.constant()));
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
  algorithm::unstable_sort(axes.begin(), axes.end(),
                           std::greater<std::size_t>());

  // ---- build new shape/view ----
  TensorShape outShape = inShape;
  TensorViewDesc outView = Xin.view();
  for (std::size_t ax : axes) {
    outShape = outShape.squeeze(ax);
    outView = outView.squeeze(ax);
  }

  HostTensor out = Xin.withView(std::move(outShape), std::move(outView));
  return {Tensor::Host(std::move(out))};
}
} // namespace denox::onnx::details::ops
