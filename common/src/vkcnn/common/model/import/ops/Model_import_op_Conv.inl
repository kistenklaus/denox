#pragma once

#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include <cassert>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Conv(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {
  // ---- arity ----
  if (inputs.size() != 2 && inputs.size() != 3) {
    throw std::runtime_error(
        fmt::format("vkcnn: Conv \"{}\": expects 2 or 3 inputs, got {}.",
                    node.name(), inputs.size()));
  }
  if (outputCount != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Conv \"{}\": must have exactly 1 output.", node.name()));
  }

  // ---- inputs ----
  if (!inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error("vkcnn: Conv: inputs X and W are required.");

  const Tensor &X = *inputs[0];
  const Tensor &W = *inputs[1];

  std::optional<Tensor> B;
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
                      node.name(), AttributeKind_name(it->second.kind())));
    if (it->second.i() != 1)
      throw std::runtime_error("vkcnn: Conv: only group=1 is supported.");
  }

  // ---- attributes: dilations ----
  glm::uvec2 dilations(1, 1);
  if (auto it = attributes.find("dilations"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": dilations must be ints, got {}.",
                      node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": dilations must have size 2 (H,W), got {}.",
          node.name(), v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: dilations must be >= 1.");
    dilations.y = static_cast<unsigned>(v[0]); // H
    dilations.x = static_cast<unsigned>(v[1]); // W
  }

  // ---- attributes: strides ----
  glm::uvec2 strides(1, 1);
  if (auto it = attributes.find("strides"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": strides must be ints, got {}.",
                      node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": strides must have size 2 (H,W), got {}.",
          node.name(), v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: strides must be >= 1.");
    strides.y = static_cast<unsigned>(v[0]); // H
    strides.x = static_cast<unsigned>(v[1]); // W
  }

  // ---- attributes: kernel_shape (optional cross-check) ----
  std::optional<glm::uvec2> kernelShapeAttr;
  if (auto it = attributes.find("kernel_shape"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": kernel_shape must be ints, got {}.",
                      node.name(), AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv \"{}\": kernel_shape must have size 2 (H,W), got {}.",
          node.name(), v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: Conv: kernel_shape must be >= 1.");
    kernelShapeAttr = glm::uvec2(static_cast<unsigned>(v[1]),  // x=W
                                 static_cast<unsigned>(v[0])); // y=H
  }

  // ---- attributes: auto_pad + pads ----
  AutoPadMode autoPad = AutoPadMode::None; // default NOTSET
  if (auto it = attributes.find("auto_pad"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(
          fmt::format("vkcnn: Conv \"{}\": auto_pad must be string, got {}.",
                      node.name(), AttributeKind_name(it->second.kind())));
    const std::string v = it->second.s();
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
                      node.name(), v));
  }

  glm::uvec2 padding(0, 0); // symmetric pad (H,W). Used only if autoPad=None.
  if (autoPad == AutoPadMode::None) {
    if (auto it = attributes.find("pads"); it != attributes.end()) {
      const Attribute &a = it->second;
      if (!a.isInts())
        throw std::runtime_error(
            fmt::format("vkcnn: Conv \"{}\": pads must be ints, got {}.",
                        node.name(), AttributeKind_name(a.kind())));
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
            "vkcnn: Conv \"{}\": pads must have size 2 or 4, got {}.",
            node.name(), v.size()));
      }
    }
  } // else: pads present but ignored under SAME_* / VALID

  // ---- weights ----
  HostTensor Wc = W.host().contiguous();
  const Dtype wdt = Wc.type();
  auto wft = dtype_to_float_type(wdt);
  if (!wft)
    throw std::runtime_error(fmt::format(
        "vkcnn: Conv: W has unsupported type {}.", dtype_to_string(wdt)));

  const TensorShape Wshape = Wc.shape(); // [K,C,R,S] (OIHW)
  if (Wshape.rank() != 4)
    throw std::runtime_error("vkcnn: Conv: W must be rank-4 (KCRS/OIHW).");

  const Symbolic K = Wshape[0], C = Wshape[1], R = Wshape[2], S = Wshape[3];
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

  // Check input channels
  if (Xdev.handle().channels() != c) {
    throw std::runtime_error(
        fmt::format("vkcnn: Conv: input channels ({}) do not match W.C ({}).",
                    Xdev.handle().channels(), c));
  }

  FilterDescriptor W_desc{
      .shape = FilterShape{s, r, c, k},    // {W, H, C, K}
      .layout = vkcnn::FilterLayout::KCRS, // OIHW
      .type = *wft,
  };
  FilterHostTensorConstView W_view{
      W_desc,
      reinterpret_cast<const std::byte *>(Wc.data()),
  };

  // ---- bias ----
  std::optional<BiasHostTensorConstView> B_view;
  if (B) {
    HostTensor Bc = B->host().contiguous();
    if (Bc.shape().rank() != 1)
      throw std::runtime_error("vkcnn: Conv: B must be 1-D [K].");
    const Symbolic KB = Bc.shape()[0];
    if (!KB.isConstant() || static_cast<unsigned>(KB.constant()) != k)
      throw std::runtime_error("vkcnn: Conv: B length must equal K.");
    const Dtype bdt = Bc.type();
    auto bft = dtype_to_float_type(bdt);
    if (!bft)
      throw std::runtime_error(fmt::format(
          "vkcnn: Conv: B has unsupported type {}.", dtype_to_string(bdt)));
    if (*bft != *wft)
      throw std::runtime_error("vkcnn: Conv: W/B float types must match.");

    BiasDescriptor B_desc{
        .shape = k,
        .layout = vkcnn::BiasLayout::C,
        .type = *bft,
    };
    B_view = BiasHostTensorConstView(
        B_desc, reinterpret_cast<const std::byte *>(Bc.data()));
  }

  // ---- SAME_* support policy (only size-preserving, symmetric, s=1, d=1) ----
  std::optional<glm::uvec2> padOpt; // passed only when autoPad=None
  if (autoPad == AutoPadMode::None) {
    padOpt = padding;
  } else if (autoPad == AutoPadMode::Zero) {
    // VALID: no padding; backend will use 0 implicitly.
    padOpt = std::nullopt;
  } else {
    // SAME_UPPER / SAME_LOWER
    if (strides != glm::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: Conv SAME_* only supported for stride=(1,1)");
    }
    if (dilations != glm::uvec2(1, 1)) {
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
    Sym Hout = state.symGraph->pool(Hin, r, py, 1,
                                    1); // stride=1, dil=1, pad=symmetric
    Sym Wout = state.symGraph->pool(Win, s, px, 1, 1);

    if (!(state.symGraph->resolve(Hout) == state.symGraph->resolve(Hin)) ||
        !(state.symGraph->resolve(Wout) == state.symGraph->resolve(Win))) {
      throw std::runtime_error("vkcnn: Conv SAME_* currently supported only "
                               "when output extent equals input extent.");
    }

    // Let backend infer pads from autoPad; we don't pass explicit padding.
    padOpt = std::nullopt;
  }

  // ---- backend ----
  vkcnn::Tensor out =
      state.output.conv2d(Xdev.handle(), W_view, B_view,
                          autoPad, // AutoPadMode
                          strides,
                          padOpt, // optional symmetric padding only when None
                          dilations
                          // atype = std::nullopt (let backend decide / W type)
      );

  return {Tensor::Device(DeviceTensor{Xdev.rank(), std::move(out)})};
}

} // namespace vkcnn::details
