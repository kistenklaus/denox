#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_ConstantOfShape(
    [[maybe_unused]] ImportState & state,
    std::span<const std::optional<Tensor>> inputs, std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // arity
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\" expects exactly 1 input (shape).",
        node.name()));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: ConstantOfShape \"{}\" must have exactly 1 output.",
                    node.name()));

  const Tensor &in = *inputs[0];
  if (in.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": runtime tensors not supported.",
        node.name()));
  const HostTensor &shapeHT = in.host();

  // shape must be INT64, rank-1, constant
  if (shapeHT.type() != Dtype::Int64)
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": input shape tensor must be INT64.",
        node.name()));
  if (shapeHT.rank() != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": input shape tensor must be rank-1.",
        node.name()));
  if (!shapeHT.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": dynamic shape not supported.",
        node.name()));

  // read extents
  HostTensor shapeC = shapeHT.contiguous();
  const auto *d64 = shapeC.storage()->i64().data();
  const std::size_t r = shapeC.storage()->i64().size();
  std::vector<std::uint64_t> extents(r);
  for (std::size_t i = 0; i < r; ++i) {
    const std::int64_t v = d64[i];
    if (v < 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": negative dimension {} at axis {}.",
          node.name(), v, i));
    extents[i] = static_cast<std::uint64_t>(v);
  }

  // output shape
  const auto g = shapeHT.shape().graph();
  assert(g != nullptr);
  assert(g.get() == state.symGraph.get());
  std::vector<Symbolic> outDims;
  outDims.reserve(r);
  for (auto e : extents)
    outDims.emplace_back(g, Sym::Const(static_cast<std::int64_t>(e)));
  TensorShape outShape{g, std::move(outDims)};

  // determine dtype from 'value' (default float32 zeros)
  Dtype outDtype = Dtype::Float32;
  std::optional<HostTensor> value;
  if (auto it = attributes.find("value"); it != attributes.end()) {
    if (!it->second.isTensor())
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must be a tensor.",
          node.name()));
    const HostTensor &v = it->second.t();
    if (!v.isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must be constant.",
          node.name()));
    if (v.sizeElemsIfStatic() != 1)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must have exactly 1 element.",
          node.name()));
    if (v.type() == Dtype::Sym)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' cannot be symbolic.",
          node.name()));
    value = v.contiguous();
    outDtype = value->type();
  }

  // total elements
  std::size_t total = 1;
  for (auto e : extents)
    total *= static_cast<std::size_t>(e);

  // zero-sized tensor
  if (total == 0) {
    std::shared_ptr<HostTensorStorage> store =
        (outDtype == Dtype::String)
            ? std::make_shared<HostTensorStorage>(
                  HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0))
            : std::make_shared<HostTensorStorage>(
                  HostTensorStorage::Raw(outDtype, nullptr, 0));
    return {Tensor::Host(HostTensor(outShape, std::move(store)))};
  }

  // build storage per dtype (no templates, no cross-type casts)
  std::shared_ptr<HostTensorStorage> store;

  switch (outDtype) {
  case Dtype::Bool: {
    if (!value.has_value() || value->type() != Dtype::Bool)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": BOOL output requires 'value' BOOL.",
          node.name()));
    const bool v = !value->storage()->boolean().empty() &&
                   value->storage()->boolean()[0] != 0;
    std::unique_ptr<bool[]> tmp(new bool[total]);
    for (std::size_t i = 0; i < total; ++i)
      tmp[i] = v;
    store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Bool(std::span<const bool>(tmp.get(), total)));
    break;
  }

  case Dtype::String: {
    if (!value.has_value() || value->type() != Dtype::String)
      throw std::runtime_error(
          fmt::format("vkcnn: ConstantOfShape \"{}\": STRING output requires "
                      "'value' STRING.",
                      node.name()));
    const char *s = value->storage()->strs()[0];
    std::vector<std::string> buf(total, s ? std::string(s) : std::string());
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::String(
        std::span<const std::string>(buf.data(), buf.size())));
    break;
  }

  case Dtype::Int8: {
    std::int8_t v = value ? value->storage()->i8()[0] : 0;
    std::vector<std::int8_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int8(
        std::span<const std::int8_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Int16: {
    std::int16_t v = value ? value->storage()->i16()[0] : 0;
    std::vector<std::int16_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int16(
        std::span<const std::int16_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Int32: {
    std::int32_t v = value ? value->storage()->i32()[0] : 0;
    std::vector<std::int32_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int32(
        std::span<const std::int32_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Int64: {
    std::int64_t v = value ? value->storage()->i64()[0] : 0;
    std::vector<std::int64_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int64(
        std::span<const std::int64_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Uint8: {
    std::uint8_t v = value ? value->storage()->u8()[0] : 0;
    std::vector<std::uint8_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint8(
        std::span<const std::uint8_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Uint16: {
    std::uint16_t v = value ? value->storage()->u16()[0] : 0;
    std::vector<std::uint16_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint16(
        std::span<const std::uint16_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Uint32: {
    std::uint32_t v = value ? value->storage()->u32()[0] : 0;
    std::vector<std::uint32_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(
        std::span<const std::uint32_t>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Uint64: {
    std::uint64_t v = value ? value->storage()->u64()[0] : 0;
    std::vector<std::uint64_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint64(
        std::span<const std::uint64_t>(buf.data(), buf.size())));
    break;
  }

  case Dtype::Float16: {
    vkcnn::f16 v{};
    if (value)
      v = value->storage()->f16()[0];
    std::vector<vkcnn::f16> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F16(
        std::span<const vkcnn::f16>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Float32: {
    vkcnn::f32 v = value ? value->storage()->f32()[0] : 0.0f;
    std::vector<vkcnn::f32> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F32(
        std::span<const vkcnn::f32>(buf.data(), buf.size())));
    break;
  }
  case Dtype::Float64: {
    vkcnn::f64 v = value ? value->storage()->f64()[0] : 0.0;
    std::vector<vkcnn::f64> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F64(
        std::span<const vkcnn::f64>(buf.data(), buf.size())));
    break;
  }

  case Dtype::Undefined:
  case Dtype::Sym:
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": unsupported dtype for fill.",
        node.name()));
  }

  HostTensor out(outShape, std::move(store));
  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
