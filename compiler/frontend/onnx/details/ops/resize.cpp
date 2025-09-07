#include "frontend/onnx/details/ops/ops.hpp"

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
        "vkcnn: Resize \"{}\" must have exactly 1 output.", nodeName));
  if (inputs.size() < 3 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\" expects at least X, roi, scales (inputs 0..2).",
        nodeName));

  // ---- attributes we support/refuse ----
  // mode: must be nearest (if provided; default usually nearest for exporters
  // we care about)
  if (auto it = attributes.find("mode"); it != attributes.end()) {
    if (!it->second.isString() || it->second.s() != "nearest")
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": only mode=\"nearest\" is supported.",
          nodeName));
  }
  // coordinate_transformation_mode: require asymmetric if provided
  if (auto it = attributes.find("coordinate_transformation_mode");
      it != attributes.end()) {
    if (!it->second.isString() || it->second.s() != "asymmetric")
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": only "
          "coordinate_transformation_mode=\"asymmetric\" is supported.",
          nodeName));
  }
  // antialias, cubic params, etc.: reject if present and non-default
  if (auto it = attributes.find("antialias"); it != attributes.end()) {
    if (!it->second.isInt() || it->second.i() != 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": antialias not supported.", nodeName));
  }

  // ---- inputs ----
  const Tensor &X = *inputs[0];
  if (!X.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": only DeviceTensor input is supported.",
        nodeName));
  const DeviceTensor &Xd = X.device();

  // roi (input[1]) must be absent or an empty tensor if provided
  if (inputs.size() >= 2 && inputs[1].has_value()) {
    const Tensor &roiT = *inputs[1];
    if (roiT.isDevice())
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": roi must be a host empty tensor if provided.",
          nodeName));
    const HostTensor &roi = roiT.host();
    if (!roi.isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": roi must be constant (and empty) if provided.",
          nodeName));
    if (roi.sizeElemsIfStatic() != 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: Resize \"{}\": non-empty roi is not supported.", nodeName));
  }

  // We only support the "scales" input (input[2]); if "sizes" (input[3]) is
  // provided, reject.
  if (inputs.size() >= 4 && inputs[3].has_value()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Resize \"{}\": sizes input not supported (use "
                    "scales with integer factors).",
                    nodeName));
  }
  if (!(inputs.size() >= 3 && inputs[2].has_value()))
    throw std::runtime_error(
        fmt::format("vkcnn: Resize \"{}\": missing scales input.", nodeName));

  const Tensor &scalesT = *inputs[2];
  if (scalesT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": scales must be a host tensor.", nodeName));
  const HostTensor &scalesH = scalesT.host();

  if (!scalesH.isConstant() || scalesH.type() != Dtype::Float32)
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": scales must be constant Float32.", nodeName));
  if (!scalesH.isContiguous())
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": scales must be contiguous.", nodeName));

  // ---- check rank / scales length ----
  const std::size_t rank = Xd.rank(); // 3 (CHW) or 4 (NCHW)
  if (rank != 3 && rank != 4)
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": input rank must be 3 or 4 (CHW/NCHW).",
        nodeName));

  const auto scales = scalesH.floats();
  if (scales.size() != rank)
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": scales length ({}) must equal input rank ({}).",
        nodeName, scales.size(), rank));

  // ---- validate scales per axis, derive integer factor s ----
  // NCHW: [N,C,H,W] ; CHW: [C,H,W]
  const std::size_t axH = (rank == 4) ? 2u : 1u;
  const std::size_t axW = (rank == 4) ? 3u : 2u;

  auto near_eq = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };

  // Require no scaling on non-spatial axes (N and C if present)
  if (rank == 4) {
    if (!near_eq(scales[0], 1.0f) || !near_eq(scales[1], 1.0f))
      throw std::runtime_error(
          fmt::format("vkcnn: Resize \"{}\": only spatial upsampling "
                      "supported; N and C scales must be 1.",
                      nodeName));
  } else { // rank == 3 → CHW
    if (!near_eq(scales[0], 1.0f))
      throw std::runtime_error(
          fmt::format("vkcnn: Resize \"{}\": only spatial upsampling "
                      "supported; C scale must be 1.",
                      nodeName));
  }

  const float sHf = scales[axH];
  const float sWf = scales[axW];
  if (sHf <= 0.0f || sWf <= 0.0f)
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": spatial scales must be positive.", nodeName));
  if (!near_eq(sHf, sWf))
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": only isotropic upsampling supported (H==W).",
        nodeName));

  // Must be an integer factor
  const float sRound = std::round(sHf);
  if (!near_eq(sHf, sRound))
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": spatial scale must be an integer.", nodeName));
  const unsigned int s = static_cast<unsigned int>(sRound);
  if (s < 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Resize \"{}\": integer scale must be >= 1.", nodeName));

  // ---- backend ----
  compiler::Tensor outHdl =
      state.output.upsample(Xd.handle(), s, compiler::FilterMode::Nearest);
  return {Tensor::Device(DeviceTensor{rank, std::move(outHdl)})};
}

} // namespace denox::onnx::details::ops
