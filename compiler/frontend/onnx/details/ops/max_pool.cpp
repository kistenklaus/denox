#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
max_pool(ImportState &state,
         memory::span<const memory::optional<Tensor>> inputs,
         std::size_t outputCount,
         const memory::hash_map<memory::string, Attribute> &attributes,
         [[maybe_unused]] opset_version version, memory::string_view nodeName) {

  // Arity / outputs: we only support the value output (no indices).
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: MaxPool \"{}\" expects exactly 1 input.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: MaxPool \"{}\": indices output not supported "
                    "(expected 1 output).",
                    nodeName));

  const Tensor &X = *inputs[0];
  if (!X.isDevice())
    throw std::runtime_error("vkcnn: MaxPool: only DeviceTensor is supported.");

  const DeviceTensor Xdev = X.device();
  const std::size_t r = Xdev.rank();
  if (r != 3 && r != 4)
    throw std::runtime_error(fmt::format(
        "vkcnn: MaxPool \"{}\": input must be CHW or NCHW.", nodeName));

  // --- Attributes ---

  // auto_pad: only NOTSET supported
  if (auto it = attributes.find("auto_pad"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(
          fmt::format("vkcnn: MaxPool \"{}\": auto_pad must be string, got {}.",
                      nodeName, it->second.kindName()));
    if (it->second.s() != "NOTSET")
      throw std::runtime_error(
          "vkcnn: MaxPool: only auto_pad=\"NOTSET\" is supported.");
  }

  // ceil_mode: must be 0/false
  if (auto it = attributes.find("ceil_mode"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(
          fmt::format("vkcnn: MaxPool \"{}\": ceil_mode must be int, got {}.",
                      nodeName, it->second.kindName()));
    if (it->second.i() != 0)
      throw std::runtime_error("vkcnn: MaxPool: ceil_mode!=0 not supported.");
  }

  // storage_order: only 0 supported (row-major indices; we don't output indices
  // anyway)
  if (auto it = attributes.find("storage_order"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": storage_order must be int, got {}.", nodeName,
          it->second.kindName()));
    if (it->second.i() != 0)
      throw std::runtime_error(
          "vkcnn: MaxPool: storage_order!=0 not supported.");
  }

  // kernel_shape (required)
  memory::uvec2 kernel(0, 0);
  {
    auto it = attributes.find("kernel_shape");
    if (it == attributes.end())
      throw std::runtime_error("vkcnn: MaxPool: kernel_shape is required.");
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": kernel_shape must be ints, got {}.", nodeName,
          a.kindName()));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": kernel_shape must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: MaxPool: kernel dims must be >= 1.");
    kernel.y = static_cast<unsigned>(v[0]); // H
    kernel.x = static_cast<unsigned>(v[1]); // W
  }

  // strides (default 1,1)
  memory::uvec2 stride(1, 1);
  if (auto it = attributes.find("strides"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: MaxPool \"{}\": strides must be ints, got {}.",
                      nodeName, AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": strides must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: MaxPool: strides must be >= 1.");
    stride.y = static_cast<unsigned>(v[0]);
    stride.x = static_cast<unsigned>(v[1]);
  }

  // dilations (default 1,1)
  memory::uvec2 dilation(1, 1);
  if (auto it = attributes.find("dilations"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: MaxPool \"{}\": dilations must be ints, got {}.",
                      nodeName, AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() != 2)
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": dilations must have size 2 (H,W), got {}.",
          nodeName, v.size()));
    if (v[0] < 1 || v[1] < 1)
      throw std::runtime_error("vkcnn: MaxPool: dilations must be >= 1.");
    dilation.y = static_cast<unsigned>(v[0]);
    dilation.x = static_cast<unsigned>(v[1]);
  }

  // pads (default 0,0; support symmetric 4-int form [top,left,bottom,right])
  memory::uvec2 padding(0, 0);
  if (auto it = attributes.find("pads"); it != attributes.end()) {
    const Attribute &a = it->second;
    if (!a.isInts())
      throw std::runtime_error(
          fmt::format("vkcnn: MaxPool \"{}\": pads must be ints, got {}.",
                      nodeName, AttributeKind_name(a.kind())));
    const auto &v = a.ints();
    if (v.size() == 2) {
      if (v[0] < 0 || v[1] < 0)
        throw std::runtime_error("vkcnn: MaxPool: pads must be >= 0.");
      padding.y = static_cast<unsigned>(v[0]); // H
      padding.x = static_cast<unsigned>(v[1]); // W
    } else if (v.size() == 4) {
      if (v[0] != v[2] || v[1] != v[3])
        throw std::runtime_error(
            "vkcnn: MaxPool: asymmetric pads not supported (require "
            "top==bottom and left==right).");
      if (v[0] < 0 || v[1] < 0)
        throw std::runtime_error("vkcnn: MaxPool: pads must be >= 0.");
      padding.y = static_cast<unsigned>(v[0]); // top/bottom
      padding.x = static_cast<unsigned>(v[1]); // left/right
    } else {
      throw std::runtime_error(fmt::format(
          "vkcnn: MaxPool \"{}\": pads must have size 2 or 4, got {}.",
          nodeName, v.size()));
    }
  }

  // Backend call
  compiler::Tensor outHandle =
      state.output.pool(Xdev.handle(), kernel, padding, stride, dilation,
                        compiler::PoolFunction::Max);
  return {Tensor::Device(DeviceTensor{Xdev.rank(), std::move(outHandle)})};
}

} // namespace denox::onnx::details::ops
