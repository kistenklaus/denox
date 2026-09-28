#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
resize(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
       std::size_t outputCount,
       const memory::hash_map<memory::string, Attribute> &attributes,
       [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // ---- arity / outputs ----
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "Resize \"{}\" must have exactly 1 output.", nodeName));
  if (inputs.size() < 3 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "Resize \"{}\" expects at least X, roi, scales (inputs 0..2).",
        nodeName));

  // ---- attributes we support/refuse ----
  // mode: nearest (default) or linear
  FilterMode filterMode = FilterMode::Nearest;
  if (auto it = attributes.find("mode"); it != attributes.end()) {
    if (it->second.isString() && it->second.s() == "linear")
      filterMode = FilterMode::Bilinear;
    else if (!it->second.isString() || it->second.s() != "nearest")
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": only mode=\"nearest\" and mode=\"linear\" are "
          "supported.",
          nodeName));
  }
  // coordinate_transformation_mode: asymmetric for nearest; half_pixel
  // (== pytorch_half_pixel for integer scales) or align_corners for linear
  if (auto it = attributes.find("coordinate_transformation_mode");
      it != attributes.end()) {
    const Attribute &ctm = it->second;
    if (filterMode == FilterMode::Nearest) {
      if (!ctm.isString() || ctm.s() != "asymmetric")
        throw std::runtime_error(fmt::format(
            "Resize \"{}\": mode=\"nearest\" only supports "
            "coordinate_transformation_mode=\"asymmetric\".",
            nodeName));
    } else if (ctm.isString() && ctm.s() == "align_corners") {
      filterMode = FilterMode::BilinearAlignCorners;
    } else if (!ctm.isString() || (ctm.s() != "half_pixel" &&
                                   ctm.s() != "pytorch_half_pixel")) {
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": mode=\"linear\" only supports "
          "coordinate_transformation_mode=\"half_pixel\", "
          "\"pytorch_half_pixel\" or \"align_corners\".",
          nodeName));
    }
  }
  // antialias, cubic params, etc.: reject if present and non-default
  if (auto it = attributes.find("antialias"); it != attributes.end()) {
    if (!it->second.isInt() || it->second.i() != 0)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": antialias not supported.", nodeName));
  }

  // ---- inputs ----
  const Tensor &X = *inputs[0];
  if (!X.isDevice())
    throw std::runtime_error(fmt::format(
        "Resize \"{}\": only DeviceTensor input is supported.",
        nodeName));
  const DeviceTensor &Xd = X.device();

  // roi (input[1]) must be absent or an empty tensor if provided
  if (inputs.size() >= 2 && inputs[1].has_value()) {
    const Tensor &roiT = *inputs[1];
    if (roiT.isDevice())
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": roi must be a host empty tensor if provided.",
          nodeName));
    const HostTensor &roi = roiT.host();
    if (!roi.isConstant())
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": roi must be constant (and empty) if provided.",
          nodeName));
    if (roi.sizeElemsIfStatic() != 0)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": non-empty roi is not supported.", nodeName));
  }

  const std::size_t rank = Xd.rank(); // 3 (CHW) or 4 (NCHW)
  if (rank != 3 && rank != 4)
    throw std::runtime_error(fmt::format(
        "Resize \"{}\": input rank must be 3 or 4 (CHW/NCHW).",
        nodeName));

  // NCHW: [N,C,H,W] ; CHW: [C,H,W]
  const std::size_t axH = (rank == 4) ? 2u : 1u;
  const std::size_t axW = (rank == 4) ? 3u : 2u;

  // ---- integer factor per spatial axis, from sizes or scales ----
  unsigned int sH = 0;
  unsigned int sW = 0;
  if (inputs.size() >= 4 && inputs[3].has_value()) {
    if (auto it = attributes.find("keep_aspect_ratio_policy");
        it != attributes.end()) {
      if (!it->second.isString() || it->second.s() != "stretch")
        throw std::runtime_error(fmt::format(
            "Resize \"{}\": only keep_aspect_ratio_policy=\"stretch\" is "
            "supported.",
            nodeName));
    }
    const Tensor &sizesT = *inputs[3];
    if (sizesT.isDevice())
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": sizes must be a host tensor.", nodeName));
    const HostTensor &sizesH = sizesT.host();
    if (!sizesH.isConstant() || sizesH.rank() != 1 ||
        sizesH.sizeElemsIfStatic() != rank)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": sizes length must equal input rank ({}).",
          nodeName, rank));
    auto size = [&](std::uint64_t i) {
      return sizesH.loadSym(memory::span<const std::uint64_t>(&i, 1));
    };

    const compiler::TensorHandle &hdl = Xd.handle();
    if ((rank == 4 && size(0) != Sym::Const(1)) ||
        size(rank - 3) != hdl.channels())
      throw std::runtime_error(
          fmt::format("Resize \"{}\": only spatial upsampling supported; N "
                      "and C sizes must match the input.",
                      nodeName));

    const SymGraph &g = *state.symGraph;
    auto factor = [&](Sym out, Sym in) -> unsigned int {
      out = g.resolve(out);
      in = g.resolve(in);
      if (!out.isConstant() || !in.isConstant() || in.constant() <= 0 ||
          out.constant() % in.constant() != 0)
        return 0;
      return static_cast<unsigned int>(out.constant() / in.constant());
    };
    sH = factor(size(axH), hdl.height());
    sW = factor(size(axW), hdl.width());
    if (sH == 0 || sW == 0)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": sizes must be a static integer multiple of the "
          "static input extents (use scales with integer factors instead).",
          nodeName));
  } else {
    if (!(inputs.size() >= 3 && inputs[2].has_value()))
      throw std::runtime_error(
          fmt::format("Resize \"{}\": missing scales input.", nodeName));

    const Tensor &scalesT = *inputs[2];
    if (scalesT.isDevice())
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": scales must be a host tensor.", nodeName));
    const HostTensor &scalesH = scalesT.host();

    if (!scalesH.isConstant() || scalesH.type() != Dtype::Float32)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": scales must be constant Float32.", nodeName));
    if (!scalesH.isContiguous())
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": scales must be contiguous.", nodeName));

    const auto scales = scalesH.floats();
    if (scales.size() != rank)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": scales length ({}) must equal input rank ({}).",
          nodeName, scales.size(), rank));

    auto near_eq = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };

    // Require no scaling on non-spatial axes (N and C if present)
    if (rank == 4) {
      if (!near_eq(scales[0], 1.0f) || !near_eq(scales[1], 1.0f))
        throw std::runtime_error(
            fmt::format("Resize \"{}\": only spatial upsampling "
                        "supported; N and C scales must be 1.",
                        nodeName));
    } else { // rank == 3 → CHW
      if (!near_eq(scales[0], 1.0f))
        throw std::runtime_error(
            fmt::format("Resize \"{}\": only spatial upsampling "
                        "supported; C scale must be 1.",
                        nodeName));
    }

    auto factor = [&](float f) -> unsigned int {
      const float r = std::round(f);
      if (r < 1.0f || !near_eq(f, r))
        return 0;
      return static_cast<unsigned int>(r);
    };
    sH = factor(scales[axH]);
    sW = factor(scales[axW]);
    if (sH == 0 || sW == 0)
      throw std::runtime_error(fmt::format(
          "Resize \"{}\": spatial scales must be positive integers.",
          nodeName));
  }
  if (sH != sW)
    throw std::runtime_error(fmt::format(
        "Resize \"{}\": only isotropic upsampling supported (H==W).",
        nodeName));

  // ---- backend ----
  compiler::TensorHandle outHdl =
      state.output.upsample(Xd.handle(), sH, filterMode);
  return {Tensor::Device(DeviceTensor{rank, std::move(outHdl)})};
}

} // namespace denox::onnx::details::ops
