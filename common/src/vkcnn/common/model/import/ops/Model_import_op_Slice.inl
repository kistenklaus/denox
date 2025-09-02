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
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> & /*attributes*/,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {
  // ---- arity ----
  if (inputs.size() < 3 || inputs.size() > 5)
    throw std::runtime_error(
        fmt::format("vkcnn: Slice \"{}\" expects 3..5 inputs (data, starts, "
                    "ends, [axes], [steps]).",
                    node.name()));
  if (!inputs[0].has_value() || !inputs[1].has_value() ||
      !inputs[2].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": data/starts/ends are required.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];
  const Tensor &startsT = *inputs[1];
  const Tensor &endsT = *inputs[2];

  const bool hasAxes = (inputs.size() >= 4) && inputs[3].has_value();
  const bool hasSteps = (inputs.size() >= 5) && inputs[4].has_value();

  // ---- helpers to read 1-D host tensors (respecting view/offset) ----
  auto read_len_1d = [](const HostTensor &t) -> std::size_t {
    if (!t.isConstant())
      throw std::runtime_error(
          "vkcnn: Slice: control tensor must have constant shape.");
    const auto dims = t.shape().toU64();
    if (dims.size() != 1)
      throw std::runtime_error("vkcnn: Slice: control tensor must be 1-D.");
    return static_cast<std::size_t>(dims[0]);
  };

  auto read_i64_1d = [&](const HostTensor &t) -> std::vector<std::int64_t> {
    if (t.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": expected INT64 tensor.", node.name()));
    const std::size_t n = read_len_1d(t);

    std::vector<std::int64_t> out(n);
    if (t.isContiguous() && t.view().offset().isConstant() &&
        t.view().offset().constant() == 0) {
      auto s = t.storage()->i64();
      if (s.size() < n)
        throw std::runtime_error(
            "vkcnn: Slice: storage smaller than logical size.");
      std::memcpy(out.data(), s.data(), n * sizeof(std::int64_t));
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t ui = static_cast<std::uint64_t>(i);
        const std::array<std::uint64_t, 1> idx{ui};
        const std::size_t elem = t.view().constIndexOf(idx);
        out[i] = t.storage()->i64()[elem];
      }
    }
    return out;
  };

  auto read_sym_1d = [&](const HostTensor &t) -> std::vector<Sym> {
    const std::size_t n = read_len_1d(t);
    std::vector<Sym> out(n);

    if (t.type() == Dtype::Sym) {
      if (t.isContiguous() && t.view().offset().isConstant() &&
          t.view().offset().constant() == 0) {
        auto s = t.storage()->sym();
        if (s.size() < n)
          throw std::runtime_error(
              "vkcnn: Slice: storage smaller than logical size.");
        for (std::size_t i = 0; i < n; ++i)
          out[i] = s[i];
      } else {
        for (std::size_t i = 0; i < n; ++i) {
          const std::uint64_t ui = static_cast<std::uint64_t>(i);
          const std::array<std::uint64_t, 1> idx{ui};
          const std::size_t elem = t.view().constIndexOf(idx);
          out[i] = t.storage()->sym()[elem];
        }
      }
    } else if (t.type() == Dtype::Int64) {
      auto ints = read_i64_1d(t);
      for (std::size_t i = 0; i < n; ++i)
        out[i] = Sym::Const(ints[i]);
    } else {
      throw std::runtime_error(
          fmt::format("vkcnn: Slice \"{}\": starts/ends must be INT64 or SYM.",
                      node.name()));
    }
    return out;
  };

  if (!startsT.isHost() || !endsT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": 'starts' and 'ends' must be host tensors.",
        node.name()));

  const HostTensor &startsH = startsT.host();
  const HostTensor &endsH = endsT.host();

  const std::vector<Sym> startsSym = read_sym_1d(startsH);
  const std::vector<Sym> endsSym = read_sym_1d(endsH);
  if (startsSym.size() != endsSym.size())
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": 'starts' and 'ends' must have same length.",
        node.name()));

  std::vector<std::int64_t> axes;
  if (hasAxes) {
    const Tensor &axesT = *inputs[3];
    if (!axesT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": 'axes' must be a host tensor.", node.name()));
    axes = read_i64_1d(axesT.host());
  }

  std::vector<std::int64_t> stepsI64; // keep integers as requested
  if (hasSteps) {
    const Tensor &stepsT = *inputs[4];
    if (!stepsT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": 'steps' must be a host tensor.", node.name()));
    stepsI64 = read_i64_1d(stepsT.host());
  }

  const std::size_t nSpec = startsSym.size();
  if (hasAxes && axes.size() != nSpec)
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": 'axes' length ({}) must match starts/ends ({}).",
        node.name(), axes.size(), nSpec));
  if (hasSteps && stepsI64.size() != nSpec)
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": 'steps' length ({}) must match starts/ends ({}).",
        node.name(), stepsI64.size(), nSpec));

  // ---- normalize axes ----
  auto normalize_axes = [&](std::size_t rank) -> std::vector<std::size_t> {
    std::vector<std::size_t> out;
    out.reserve(nSpec);
    if (hasAxes) {
      for (auto a : axes) {
        if (a < 0)
          a += static_cast<std::int64_t>(rank);
        if (a < 0 || a >= static_cast<std::int64_t>(rank))
          throw std::runtime_error(fmt::format(
              "vkcnn: Slice \"{}\": axis {} out of range for rank {}.",
              node.name(), a, rank));
        out.push_back(static_cast<std::size_t>(a));
      }
    } else {
      for (std::size_t i = 0; i < nSpec; ++i)
        out.push_back(i);
    }
    return out;
  };

  // ===========================================================================
  // DEVICE PATH (crop only H/W; steps must be +1)
  // ===========================================================================
  if (dataT.isDevice()) {
    const DeviceTensor &devIn = dataT.device();
    const std::size_t r = devIn.rank(); // 3 (CHW) or 4 (NCHW)
    if (r != 3 && r != 4)
      throw std::runtime_error(
          fmt::format("vkcnn: Slice \"{}\": device tensor must be CHW or NCHW.",
                      node.name()));

    const auto ax = normalize_axes(r);

    // default steps → 1
    std::vector<std::int64_t> stepsDev(nSpec, 1);
    if (hasSteps)
      stepsDev = stepsI64;

    for (std::size_t i = 0; i < nSpec; ++i) {
      if (stepsDev[i] != 1)
        throw std::runtime_error(fmt::format(
            "vkcnn: Slice \"{}\": device path supports only step==1.",
            node.name()));
    }

    auto g = state.symGraph;

    // extents (as Sym) of input (symbolic allowed)
    const Sym H = devIn.handle().height().resolve();
    const Sym W = devIn.handle().width().resolve();

    // default full-pass on H/W
    Sym left = Sym::Const(0);
    Sym right = W;
    Sym top = Sym::Const(0);
    Sym bot = H;

    auto norm_index_if_const_neg = [&](Sym idx, Sym dim) -> Sym {
      if (idx.isConstant() && idx.constant() < 0) {
        return g->add(dim, idx); // dim + (negative)
      }
      return idx;
    };

    for (std::size_t i = 0; i < nSpec; ++i) {
      const std::size_t a = ax[i];
      const Sym s = startsSym[i];
      const Sym e = endsSym[i];

      if (r == 4) {
        if (a == 2) { // H
          top = norm_index_if_const_neg(s, H);
          bot = norm_index_if_const_neg(e, H);
        } else if (a == 3) { // W
          left = norm_index_if_const_neg(s, W);
          right = norm_index_if_const_neg(e, W);
        } else {
          // N or C → must be no-op on 'start'
          if (!(s.isConstant() && s.constant() == 0))
            throw std::runtime_error(
                fmt::format("vkcnn: Slice \"{}\": device slicing supports only "
                            "H/W. N/C must be no-op.",
                            node.name()));
        }
      } else {        // r == 3 (CHW)
        if (a == 1) { // H
          top = norm_index_if_const_neg(s, H);
          bot = norm_index_if_const_neg(e, H);
        } else if (a == 2) { // W
          left = norm_index_if_const_neg(s, W);
          right = norm_index_if_const_neg(e, W);
        } else { // C
          if (!(s.isConstant() && s.constant() == 0))
            throw std::runtime_error(
                fmt::format("vkcnn: Slice \"{}\": device slicing supports only "
                            "H/W. C must be no-op.",
                            node.name()));
        }
      }
    }

    // Delegate exact clamping to backend; hypergraph will carry
    // right-left/top-bottom. (No min/max symbols introduced here.)
    vkcnn::Tensor outH =
        state.output.slice(devIn.handle(), left, right, top, bot);
    DeviceTensor outDev(r, std::move(outH));
    return {Tensor::Device(std::move(outDev))};
  }

  // ===========================================================================
  // HOST PATH (general N-D; supports negative steps and reversing)
  // starts/ends must be INT64 here (not Sym).
  // ===========================================================================
  if (startsH.type() == Dtype::Sym || endsH.type() == Dtype::Sym)
    throw std::runtime_error(fmt::format(
        "vkcnn: Slice \"{}\": host path does not support SYM starts/ends.",
        node.name()));

  const std::vector<std::int64_t> startsI64 = read_i64_1d(startsH);
  const std::vector<std::int64_t> endsI64 = read_i64_1d(endsH);
  std::vector<std::int64_t> stepsHost(nSpec, 1);
  if (hasSteps)
    stepsHost = stepsI64;

  const HostTensor &Xin = dataT.host();
  const std::size_t r = Xin.rank();
  const auto ax = normalize_axes(r);

  auto g = Xin.shape().graph();
  TensorViewDesc view = Xin.view();
  TensorShape shape = Xin.shape();

  // We guarantee non-negative constant lengths, without introducing min/max
  // symbols.
  for (std::size_t i = 0; i < nSpec; ++i) {
    const std::size_t a = ax[i];
    const std::int64_t step = stepsHost[i];
    if (step == 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: Slice \"{}\": step must be non-zero.", node.name()));

    // Host tensor shapes are constant; assert that and use ints for clamping.
    if (!shape[a].isConstant())
      throw std::runtime_error(
          "vkcnn: Slice host path expects constant shape.");
    int64_t dimC = shape[a].constant();

    int64_t st = startsI64[i];
    int64_t en = endsI64[i];

    if (step > 0) {
      // Python-like normalize and clamp to [0, dimC]
      if (st < 0)
        st += dimC;
      if (en < 0)
        en += dimC;
      if (st < 0)
        st = 0;
      else if (st > dimC)
        st = dimC;
      if (en < 0)
        en = 0;
      else if (en > dimC)
        en = dimC;

      int64_t len = 0;
      if (en > st) {
        len = (en - st + (step - 1)) / step; // ceil_div
      }
      // Update view/shape
      view = view.slice(a, Symbolic(g, Sym::Const(st)),
                        Symbolic(g, Sym::Const(step)));
      shape[a] = Symbolic(g, Sym::Const(len));
    } else {
      const int64_t ns = -step;
      if (st < 0)
        st += dimC;
      if (en < 0)
        en += dimC;
      // Clamp to valid inclusive bounds for negative traversal
      if (st < -1)
        st = -1;
      else if (st >= dimC)
        st = dimC - 1;
      if (en < -1)
        en = -1;
      else if (en >= dimC)
        en = dimC - 1;

      int64_t len = 0;
      if (st > en) {
        len = (st - en + (ns - 1)) / ns; // ceil_div
      }

      // Represent descending order with a negative stride view (no extra
      // reverse).
      view = view.slice(a, Symbolic(g, Sym::Const(st)),
                        Symbolic(g, Sym::Const(step)));
      shape[a] = Symbolic(g, Sym::Const(len));
    }
  }

  // Normalize any negative strides left (should be none after reverse, but
  // safe)

  HostTensor out = Xin.withView(std::move(shape), std::move(view));
  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
