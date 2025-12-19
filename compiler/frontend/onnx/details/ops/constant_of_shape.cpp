#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
constant_of_shape([[maybe_unused]] ImportState &state,
                  memory::span<const memory::optional<Tensor>> inputs,
                  std::size_t outputCount,
                  const memory::hash_map<memory::string, Attribute> &attributes,
                  [[maybe_unused]] opset_version version,
                  memory::string_view nodeName) {

  // arity
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\" expects exactly 1 input (shape).",
        nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &in = *inputs[0];
  if (in.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": runtime tensors not supported.",
        nodeName));
  const HostTensor &shapeHT = in.host();

  // shape must be INT64, rank-1, constant
  if (shapeHT.type() != Dtype::Int64)
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": input shape tensor must be INT64.",
        nodeName));
  if (shapeHT.rank() != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": input shape tensor must be rank-1.",
        nodeName));
  if (!shapeHT.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": dynamic shape not supported.",
        nodeName));

  // read extents
  HostTensor shapeC = shapeHT.contiguous();
  const auto *d64 = shapeC.storage()->i64().data();
  const std::size_t r = shapeC.storage()->i64().size();
  memory::vector<std::uint64_t> extents(r);
  for (std::size_t i = 0; i < r; ++i) {
    const std::int64_t v = d64[i];
    if (v < 0)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": negative dimension {} at axis {}.",
          nodeName, v, i));
    extents[i] = static_cast<std::uint64_t>(v);
  }

  // output shape
  const auto g = state.symGraph;
  assert(g != nullptr);
  memory::vector<compiler::Symbolic> outDims;
  outDims.reserve(r);
  for (auto e : extents)
    outDims.emplace_back(g, Sym::Const(static_cast<std::int64_t>(e)));
  TensorShape outShape{g, std::move(outDims)};

  // determine dtype from 'value' (default float32 zeros)
  Dtype outDtype = Dtype::Float32;
  memory::optional<HostTensor> value;
  if (auto it = attributes.find("value"); it != attributes.end()) {
    if (!it->second.isTensor())
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must be a tensor.",
          nodeName));
    const HostTensor &v = it->second.t();
    if (!v.isConstant())
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must be constant.",
          nodeName));
    if (v.sizeElemsIfStatic() != 1)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' must have exactly 1 element.",
          nodeName));
    if (v.type() == Dtype::Sym)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": 'value' cannot be symbolic.",
          nodeName));
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

  switch (outDtype.kind()) {
  case DtypeKind::Bool: {
    if (!value.has_value() || value->type() != Dtype::Bool)
      throw std::runtime_error(fmt::format(
          "vkcnn: ConstantOfShape \"{}\": BOOL output requires 'value' BOOL.",
          nodeName));
    const bool v = !value->storage()->boolean().empty() &&
                   value->storage()->boolean()[0] != 0;
    std::unique_ptr<bool[]> tmp(new bool[total]);
    for (std::size_t i = 0; i < total; ++i)
      tmp[i] = v;
    store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Bool(memory::span<const bool>(tmp.get(), total)));
    break;
  }

  case DtypeKind::String: {
    if (!value.has_value() || value->type() != Dtype::String)
      throw std::runtime_error(
          fmt::format("vkcnn: ConstantOfShape \"{}\": STRING output requires "
                      "'value' STRING.",
                      nodeName));
    const char *s = value->storage()->strs()[0];
    memory::vector<memory::string> buf(total, s ? memory::string(s) : memory::string());
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::String(
        memory::span<const memory::string>(buf.data(), buf.size())));
    break;
  }

  case DtypeKind::Int8: {
    std::int8_t v = value ? value->storage()->i8()[0] : 0;
    memory::vector<std::int8_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int8(
        memory::span<const std::int8_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Int16: {
    std::int16_t v = value ? value->storage()->i16()[0] : 0;
    memory::vector<std::int16_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int16(
        memory::span<const std::int16_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Int32: {
    std::int32_t v = value ? value->storage()->i32()[0] : 0;
    memory::vector<std::int32_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int32(
        memory::span<const std::int32_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Int64: {
    std::int64_t v = value ? value->storage()->i64()[0] : 0;
    memory::vector<std::int64_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int64(
        memory::span<const std::int64_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Uint8: {
    std::uint8_t v = value ? value->storage()->u8()[0] : 0;
    memory::vector<std::uint8_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint8(
        memory::span<const std::uint8_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Uint16: {
    std::uint16_t v = value ? value->storage()->u16()[0] : 0;
    memory::vector<std::uint16_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint16(
        memory::span<const std::uint16_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Uint32: {
    std::uint32_t v = value ? value->storage()->u32()[0] : 0;
    memory::vector<std::uint32_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(
        memory::span<const std::uint32_t>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Uint64: {
    std::uint64_t v = value ? value->storage()->u64()[0] : 0;
    memory::vector<std::uint64_t> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint64(
        memory::span<const std::uint64_t>(buf.data(), buf.size())));
    break;
  }

  case DtypeKind::Float16: {
    memory::f16 v{};
    if (value)
      v = value->storage()->f16()[0];
    memory::vector<memory::f16> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F16(
        memory::span<const memory::f16>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Float32: {
    memory::f32 v = value ? value->storage()->f32()[0] : 0.0f;
    memory::vector<memory::f32> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F32(
        memory::span<const memory::f32>(buf.data(), buf.size())));
    break;
  }
  case DtypeKind::Float64: {
    memory::f64 v = value ? value->storage()->f64()[0] : 0.0;
    memory::vector<memory::f64> buf(total, v);
    store = std::make_shared<HostTensorStorage>(HostTensorStorage::F64(
        memory::span<const memory::f64>(buf.data(), buf.size())));
    break;
  }

  case DtypeKind::Undefined:
  case DtypeKind::Sym:
    throw std::runtime_error(fmt::format(
        "vkcnn: ConstantOfShape \"{}\": unsupported dtype for fill.",
        nodeName));
  }

  HostTensor out(outShape, std::move(store));
  return {Tensor::Host(std::move(out))};
}

} // namespace denox::onnx::details::ops
