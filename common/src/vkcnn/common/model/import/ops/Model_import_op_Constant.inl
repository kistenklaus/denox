#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Constant(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  if (!inputs.empty())
    throw std::runtime_error(fmt::format(
        "vkcnn: Constant node \"{}\" must have 0 inputs.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Constant node \"{}\" must have exactly 1 output.",
                    node.name()));
  if (!state.symGraph)
    throw std::runtime_error("vkcnn: Constant: symGraph is null");

  const auto it_value = attributes.find("value");
  const auto it_sparse_value = attributes.find("sparse_value");
  const auto it_f = attributes.find("value_float");
  const auto it_fs = attributes.find("value_floats");
  const auto it_i = attributes.find("value_int");
  const auto it_is = attributes.find("value_ints");
  const auto it_s = attributes.find("value_string");
  const auto it_ss = attributes.find("value_strings");

  int present = 0;
  present += (it_value != attributes.end());
  present += (it_sparse_value != attributes.end());
  present += (it_f != attributes.end());
  present += (it_fs != attributes.end());
  present += (it_i != attributes.end());
  present += (it_is != attributes.end());
  present += (it_s != attributes.end());
  present += (it_ss != attributes.end());

  if (present == 0) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Constant node \"{}\" has no value attribute.", node.name()));
  }
  if (present > 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Constant node \"{}\" specifies multiple value attributes.",
        node.name()));
  }

  // 1) Tensor payload: use existing tensor importer (Host-only).
  if (it_value != attributes.end()) {
    const Attribute &a = it_value->second;
    if (!a.isTensor())
      throw std::runtime_error(fmt::format(
          "vkcnn: Constant \"{}\": attribute 'value' is not a tensor.",
          node.name()));
    HostTensor ht = a.t(); // Already a HostTensor
    return {Tensor::Host(std::move(ht))};
  }

  // 2) Sparse tensor: unsupported.
  if (it_sparse_value != attributes.end()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Constant \"{}\": sparse_value not supported.", node.name()));
  }

  // Helpers to build HostTensor from scalars / vectors.
  const auto &g = state.symGraph;

  auto make_scalar_shape = [&]() -> TensorShape {
    std::vector<Symbolic> dims; // rank-0 (scalar)
    return TensorShape{g, std::move(dims)};
  };
  auto make_1d_shape = [&](std::size_t n) -> TensorShape {
    std::vector<Symbolic> dims;
    dims.emplace_back(g, Sym::Const(static_cast<int64_t>(n)));
    return TensorShape{g, std::move(dims)};
  };

  // 3) value_float → scalar float32
  if (it_f != attributes.end()) {
    const Attribute &a = it_f->second;
    if (!a.isFloat())
      throw std::runtime_error("vkcnn: Constant: value_float has wrong type.");
    float v = a.f();
    auto store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::F32(std::span<const float>(&v, 1)));
    return {Tensor::Host(HostTensor(make_scalar_shape(), std::move(store)))};
  }

  // 4) value_floats → 1-D float32
  if (it_fs != attributes.end()) {
    const Attribute &a = it_fs->second;
    if (!a.isFloats())
      throw std::runtime_error("vkcnn: Constant: value_floats has wrong type.");
    const auto &vec = a.floats();
    auto store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::F32(std::span<const float>(vec.data(), vec.size())));
    return {
        Tensor::Host(HostTensor(make_1d_shape(vec.size()), std::move(store)))};
  }

  // 5) value_int → scalar int64
  if (it_i != attributes.end()) {
    const Attribute &a = it_i->second;
    if (!a.isInt())
      throw std::runtime_error("vkcnn: Constant: value_int has wrong type.");
    std::int64_t v = a.i();
    auto store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Int64(std::span<const std::int64_t>(&v, 1)));
    return {Tensor::Host(HostTensor(make_scalar_shape(), std::move(store)))};
  }

  // 6) value_ints → 1-D int64
  if (it_is != attributes.end()) {
    const Attribute &a = it_is->second;
    if (!a.isInts())
      throw std::runtime_error("vkcnn: Constant: value_ints has wrong type.");
    const auto &vec = a.ints();
    auto store = std::make_shared<HostTensorStorage>(HostTensorStorage::Int64(
        std::span<const std::int64_t>(vec.data(), vec.size())));
    return {
        Tensor::Host(HostTensor(make_1d_shape(vec.size()), std::move(store)))};
  }

  // 7) value_string → scalar string
  if (it_s != attributes.end()) {
    const Attribute &a = it_s->second;
    if (!a.isString())
      throw std::runtime_error("vkcnn: Constant: value_string has wrong type.");
    const std::string &sv = a.s();
    auto store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::String(std::span<const std::string>(&sv, 1)));
    return {Tensor::Host(HostTensor(make_scalar_shape(), std::move(store)))};
  }

  // 8) value_strings → 1-D strings
  if (it_ss != attributes.end()) {
    const Attribute &a = it_ss->second;
    if (!a.isStrings())
      throw std::runtime_error(
          "vkcnn: Constant: value_strings has wrong type.");
    const auto &vec = a.strings();
    auto store = std::make_shared<HostTensorStorage>(HostTensorStorage::String(
        std::span<const std::string>(vec.data(), vec.size())));
    return {
        Tensor::Host(HostTensor(make_1d_shape(vec.size()), std::move(store)))};
  }

  // Fallback: should be unreachable because we counted 'present' above.
  throw std::logic_error("vkcnn: Constant: unreachable dispatch.");
}

} // namespace vkcnn::details
