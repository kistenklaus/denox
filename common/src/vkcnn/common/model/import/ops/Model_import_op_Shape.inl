#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Shape(ImportState &state,
                std::span<const std::optional<Tensor>> inputs,
                std::size_t outputCount,
                const std::unordered_map<std::string, Attribute> &attributes,
                opset_version version, const onnx::NodeProto &node) {

  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Shape expects exactly 1 output (node = \"{}\")", node.name()));
  if (inputs.size() != 1 || !inputs[0].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Shape expects 1 input (node = \"{}\")", node.name()));

  const Tensor &in = *inputs[0];
  const TensorShape inShape = in.shape();
  const std::size_t r = inShape.rank();

  // Optional start/end (use Python-like semantics)
  std::int64_t start = 0;
  std::int64_t end = static_cast<std::int64_t>(r);

  auto itS = attributes.find("start");
  if (itS != attributes.end()) {
    if (!itS->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: Shape: 'start' must be INT (node = \"{}\")", node.name()));
    start = itS->second.i();
  }
  auto itE = attributes.find("end");
  if (itE != attributes.end()) {
    if (!itE->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: Shape: 'end' must be INT (node = \"{}\")", node.name()));
    end = itE->second.i();
  }

  auto norm = [&](std::int64_t v) -> std::int64_t {
    if (v < 0)
      v += static_cast<std::int64_t>(r);
    if (v < 0)
      v = 0;
    if (v > static_cast<std::int64_t>(r))
      v = static_cast<std::int64_t>(r);
    return v;
  };
  const std::int64_t s = norm(start);
  const std::int64_t e = norm(end);
  const std::size_t outLen = (e > s) ? static_cast<std::size_t>(e - s) : 0;

  // Output shape is 1-D [outLen]
  std::vector<std::uint64_t> outDim = {outLen};
  TensorShape outShape(state.symGraph, std::span<const std::uint64_t>(outDim));

  std::shared_ptr<HostTensorStorage> store;

  if (inShape.isConstant()) {
    // Return INT64 (standard ONNX)
    std::vector<std::int64_t> vals;
    vals.reserve(outLen);
    for (std::size_t i = 0; i < outLen; ++i) {
      const auto &d = inShape[static_cast<std::size_t>(s) + i];
      // by isConstant(), all dims are constant
      vals.push_back(static_cast<std::int64_t>(d.constant()));
    }
    store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Int64(std::span<const std::int64_t>(vals)));
  } else {
    // Return SYM
    std::vector<Sym> syms;
    syms.reserve(outLen);
    for (std::size_t i = 0; i < outLen; ++i) {
      const auto &d = inShape[static_cast<std::size_t>(s) + i];
      syms.push_back(
          static_cast<Sym>(*d)); // Symbolic -> Sym (const or symbolic)
    }
    store = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Sym(std::span<const Sym>(syms)));
  }

  HostTensor ht(outShape, std::move(store));
  return {Tensor::Host(std::move(ht))};
}

} // namespace vkcnn::details
