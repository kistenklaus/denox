#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <cassert>
#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> conv(
    ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() != 2 && inputs.size() != 3) {
    throw std::runtime_error(
        fmt::format("vkcnn: Conv \"{}\": expects 2 or 3 inputs, got {}.",
                    nodeName, inputs.size()));
  }
  if (outputCount != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Conv \"{}\": must have exactly 1 output.", nodeName));
  }

  // ---- inputs ----
  if (!inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error("vkcnn: Conv: inputs X and W are required.");

  const Tensor &X = *inputs[0];
  const Tensor &W = *inputs[1];

  memory::optional<Tensor> B;
  if (inputs.size() == 3 && inputs[2].has_value())
    B = *inputs[2];

  if (!X.isDevice())
    throw std::runtime_error("vkcnn: Conv: X must be a DeviceTensor.");
  if (!W.isHost())
    throw std::runtime_error("vkcnn: Conv: W must be a HostTensor.");
  if (B && !B->isHost())
    throw std::runtime_error(
        "vkcnn: Conv: B (if present) must be a HostTensor.");

  const DeviceTensor Xdev = X.device();

  if (Xdev.rank() != 3 && Xdev.rank() != 4)
    throw std::runtime_error("vkcnn: Conv: X must be CHW or NCHW.");

  // ---- attributes: group ----
  if (auto it = attributes.find("group"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": group must be int, got {}.",
                      nodeName, it->second.kindName()));
    if (it->second.i() != 1)
      throw std::runtime_error("vkcnn: Conv: only group=1 is supported.");
  }

  // ---- attributes: dilations ----
  memory::uvec2 dilations(1, 1);
  if (auto it = attributes.find("dilations"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": dilations must be ints, got {}.",
                      nodeName, a.kindName()));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": dilations must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: dilations must be >= 1.");
    dilations.y = static_cast<unsigned>(v[0]); // H
    dilations.x = static_cast<unsigned>(v[1]); // W
  }

  // ---- attributes: strides ----
  memory::uvec2 strides(1, 1);
  if (auto it = attributes.find("strides"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": strides must be ints, got {}.",
                      nodeName, a.kindName()));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": strides must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: strides must be >= 1.");
    strides.y = static_cast<unsigned>(v[0]); // H
    strides.x = static_cast<unsigned>(v[1]); // W
  }

  // ---- attributes: kernel_shape (optional cross-check) ----
  memory::optional<memory::uvec2> kernelShapeAttr;
  if (auto it = attributes.find("kernel_shape"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": kernel_shape must be ints, got {}.",
                      nodeName, a.kindName()));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": kernel_shape must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: kernel_shape must be >= 1.");
    kernelShapeAttr = memory::uvec2(static_cast<unsigned>(v[1]),  // x=W
                                    static_cast<unsigned>(v[0])); // y=H
  }

  // ---- attributes: auto_pad + pads ----
  AutoPadMode autoPad = AutoPadMode::None; // default NOTSET
  if (auto it = attributes.find("auto_pad"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": auto_pad must be string, got {}.",
                      nodeName, it->second.kindName()));
    const memory::string v = it->second.s();
    if (v == "NOTSET")
      autoPad = AutoPadMode::None;
    else if (v == "SAME_UPPER")
      autoPad = AutoPadMode::SameUpper;
    else if (v == "SAME_LOWER")
      autoPad = AutoPadMode::SameLower;
    else if (v == "VALID")
      autoPad = AutoPadMode::Zero;
    else
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": unsupported auto_pad value \"{}\".",
                      nodeName, v));
  }

  memory::uvec2 padding(0,
                        0); // symmetric pad (H,W). Used only if autoPad=None.
  if (autoPad == AutoPadMode::None) {
    if (auto it = attributes.find("pads"); it != attributes.end()) {
      const Attribute &a = it->second;
      if (!a.isInts())
        throw std::runtime_error(
            fmt::format("vkcnn: Conv \"{}\": pads must be ints, got {}.",
                        nodeName, AttributeKind_name(a.kind())));
      const auto &v = a.ints();
      if (v.size() == 2) {
        // shorthand [pad_h, pad_w]
        if (v[0] < 0 || v[1] < 0)
          throw std::runtime_error("vkcnn: Conv: pads must be >= 0.");
        padding.y = static_cast<unsigned>(v[0]);
        padding.x = static_cast<unsigned>(v[1]);
      } else if (v.size() == 4) {
        // canonical [top, left, bottom, right] (must be symmetric)
        if (v[0] != v[2] || v[1] != v[3])
          throw std::runtime_error("vkcnn: Conv: asymmetric pads not supported "
                                   "(require top==bottom and left==right).");
        if (v[0] < 0 || v[1] < 0)
          throw std::runtime_error("vkcnn: Conv: pads must be >= 0.");
        padding.y = static_cast<unsigned>(v[0]); // top/bottom
        padding.x = static_cast<unsigned>(v[1]); // left/right
      } else {
        throw std::runtime_error(fmt::format(
            "vkcnn: Conv \"{}\": pads must have size 2 or 4, got {}.", nodeName,
            v.size()));
      }
    }
  } // else: pads present but ignored under SAME_* / VALID

  // ---- weights ----
  HostTensor Wc = W.host().contiguous();
  const Dtype wdt = Wc.type();
  auto wft = wdt.toDenoxType();
  if (!wft)
    throw std::runtime_error(fmt::format(
        "vkcnn: Conv: W has unsupported type {}.", wdt.to_string()));

  const TensorShape Wshape = Wc.shape(); // [K,C,R,S] (OIHW)
  if (Wshape.rank() != 4)
    throw std::runtime_error("vkcnn: Conv: W must be rank-4 (KCRS/OIHW).");

  const compiler::Symbolic K = Wshape[0], C = Wshape[1], R = Wshape[2],
                           S = Wshape[3];
  if (!K.isConstant() || !C.isConstant() || !R.isConstant() || !S.isConstant())
    throw std::runtime_error("vkcnn: Conv: W must have constant K,C,R,S.");

  const unsigned k = static_cast<unsigned>(K.constant());
  const unsigned c = static_cast<unsigned>(C.constant());
  const unsigned r = static_cast<unsigned>(R.constant());
  const unsigned s = static_cast<unsigned>(S.constant());

  if (kernelShapeAttr && (kernelShapeAttr->y != r || kernelShapeAttr->x != s))
    throw std::runtime_error(fmt::format(
        "vkcnn: Conv: kernel_shape mismatch (W has {}x{}, attr is {}x{}).", r,
        s, kernelShapeAttr->y, kernelShapeAttr->x));

  if (!Xdev.handle().channels().isConstant()) {
    throw std::runtime_error("vkcnn: Conv: input channels must be constant!");
  }



  // Check input channels
  if (Xdev.handle().channels().constant() != c) {
    throw std::runtime_error(
        fmt::format("vkcnn: Conv: input channels ({}) do not match W.C ({}).",
                    Xdev.handle().channels().constant(), c));
  }

  memory::FilterDescriptor W_desc{
      .shape = memory::FilterShape{s, r, c, k}, // {W, H, C, K}
      .layout = memory::FilterLayout::KCRS,     // OIHW
      .type = *wft,
  };
  memory::FilterTensorConstView W_view{
      W_desc,
      reinterpret_cast<const std::byte *>(Wc.data()),
  };

  // ---- bias ----
  memory::optional<memory::BiasTensorConstView> B_view;
  if (B) {
    HostTensor Bc = B->host().contiguous();
    if (Bc.shape().rank() != 1)
      throw std::runtime_error("vkcnn: Conv: B must be 1-D [K].");
    const compiler::Symbolic KB = Bc.shape()[0];
    if (!KB.isConstant() || static_cast<unsigned>(KB.constant()) != k)
      throw std::runtime_error("vkcnn: Conv: B length must equal K.");
    const Dtype bdt = Bc.type();
    auto bft = bdt.toDenoxType();
    if (!bft)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv: B has unsupported type {}.", bdt.to_string()));
    if (*bft != *wft)
      throw std::runtime_error("vkcnn: Conv: W/B float types must match.");

    memory::BiasDescriptor B_desc{
        .shape = k,
        .layout = memory::BiasLayout::C,
        .type = *bft,
    };
    B_view = memory::BiasTensorConstView(
        B_desc, reinterpret_cast<const std::byte *>(Bc.data()));
  }

  // ---- SAME_* support policy (only size-preserving, symmetric, s=1, d=1) ----
  memory::optional<memory::uvec2> padOpt; // passed only when autoPad=None
  if (autoPad == AutoPadMode::None) {
    padOpt = padding;
  } else if (autoPad == AutoPadMode::Zero) {
    // VALID: no padding; backend will use 0 implicitly.
    padOpt = memory::nullopt;
  } else {
    // SAME_UPPER / SAME_LOWER
    if (strides != memory::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: Conv SAME_* only supported for stride=(1,1)");
    }
    if (dilations != memory::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: Conv SAME_* only supported for dilation=(1,1)");
    }

    const unsigned sumX = (s > 0) ? (s - 1u) : 0u;
    const unsigned sumY = (r > 0) ? (r - 1u) : 0u;
    if ((sumX & 1u) != 0u || (sumY & 1u) != 0u) {
      // We only support symmetric SAME (size-preserving) which requires odd
      // kernels.
      throw std::runtime_error("vkcnn: Conv SAME_* requires odd kernel sizes "
                               "(enables symmetric padding).");
    }
    const unsigned px = sumX / 2u;
    const unsigned py = sumY / 2u;

    // Prove size-preserving with the symbolic engine (no min/max involved).
    Sym Hin = Xdev.handle().height().resolve();
    Sym Win = Xdev.handle().width().resolve();
    Sym Hout =
        state.symGraph->pool(Hin, r, py, 1,
                             1); // stride=1, dil=1, pad=symmetric
    Sym Wout = state.symGraph->pool(Win, s, px, 1, 1);

    if (!(state.symGraph->resolve(Hout) == state.symGraph->resolve(Hin)) ||
        !(state.symGraph->resolve(Wout) == state.symGraph->resolve(Win))) {
      throw std::runtime_error("vkcnn: Conv SAME_* currently supported only "
                               "when output extent equals input extent.");
    }

    // Let backend infer pads from autoPad; we don't pass explicit padding.
    padOpt = memory::nullopt;
  }

  // ---- backend ----
  compiler::Tensor out =
      state.output.conv2d(Xdev.handle(), W_view, B_view,
                          autoPad, // AutoPadMode
                          strides,
                          padOpt, // optional symmetric padding only when None
                          dilations
                          // atype = memory::nullopt (let backend decide / W type)
      );
  return {Tensor::Device(DeviceTensor{Xdev.rank(), std::move(out)})};
}

} // namespace denox::onnx::details::ops
