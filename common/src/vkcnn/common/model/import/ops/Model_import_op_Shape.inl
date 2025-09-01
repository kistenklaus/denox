#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <algorithm>
#include <fmt/format.h>
#include <google/protobuf/any.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Shape(
    [[maybe_unused]] ImportState &state,
    [[maybe_unused]] std::span<const std::optional<Tensor>> inputs,
    [[maybe_unused]] std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor> &attributes,
    [[maybe_unused]] opset_version version,
    [[maybe_unused]] const onnx::NodeProto &node) {

  if (inputs.size() != 1) {
    throw std::runtime_error(
        fmt::format("vkcnn: Operation Shape requires exactly one input. Got {}",
                    inputs.size()));
  }
  if (!inputs.front().has_value()) {
    throw std::runtime_error(
        "vkcnn: Operation Shape has input, but the input was not provided");
  }
  const auto &input = *inputs.front();

  if (input.isScalar() || input.isString()) {
    return {Tensor::Shape(ShapeTensor::Tensor({}))};
  }
  if (input.isUnknown()) {
    throw std::runtime_error(
        "vkcnn: Failed to infer input type of operation Shape");
  }
  if (input.isList()) {
    throw std::runtime_error(
        "vkcnn: Invalid type of input for operation Shape");
  }
  assert(input.isRaw() || input.isRuntimeTensor() || input.isShape());

  ShapeTensor shape = input.shape();
  if (shape.isScalar()) {
    return {Tensor::Shape(ShapeTensor::Tensor({}))};
  }

  unsigned int rank = shape.rank();

  std::int64_t start = 0;
  std::int64_t end = rank;

  if (version < 15) {
    if (attributes.contains("start")) {
      auto startAttrib = attributes.at("start");
      if (startAttrib.isUnknown()) {
        start = 0;
      } else {
        if (!startAttrib.isScalar()) {
          throw std::runtime_error(
              "vkcnn: Operation Shape attribute \"start\" is not a integer.");
        }
        const auto &scalarAttrib = startAttrib.scalar();
        if (scalarAttrib.dtype != Dtype::Int64 &&
            scalarAttrib.dtype != Dtype::Int16 &&
            scalarAttrib.dtype != Dtype::Int32) {
          throw std::runtime_error(
              "vkcnn: Operation Shape attribute \"start\" is not a integer.");
        }
        start = scalarAttrib.v.i;
      }
    }
    if (attributes.contains("end")) {
      auto endAttrib = attributes.at("end");
      if (endAttrib.isUnknown()) {
        end = 0;
      } else {
        if (!endAttrib.isScalar()) {
          throw std::runtime_error(
              "vkcnn: Operation Shape attribute \"end\" is not a integer.");
        }
        const auto &scalarAttrib = endAttrib.scalar();
        if (scalarAttrib.dtype != Dtype::Int64 &&
            scalarAttrib.dtype != Dtype::Int16 &&
            scalarAttrib.dtype != Dtype::Int32) {
          throw std::runtime_error(
              "vkcnn: Operation Shape attribute \"end\" is not a integer.");
        }
        end = scalarAttrib.v.i;
      }
    }
  } else {
    if (attributes.contains("start")) {
      throw std::runtime_error(
          fmt::format("vkcnn: Operation Shape (version: {}) does not support "
                      "attribute \"start\"",
                      version));
    }
    if (attributes.contains("end")) {
      throw std::runtime_error(
          fmt::format("vkcnn: Operation Shape (version: {}) does not support "
                      "attribute \"end\"",
                      version));
    }
  }
  // NOTE: Rank is always known at compiletime, we do not support unknown
  // shapes.
  if (start < 0) {
    start = rank + start;
  }
  if (end < 0) {
    end = rank + end;
  }

  end = std::clamp<std::int64_t>(end, 0, rank);
  start = std::clamp<std::int64_t>(start, 0, rank);
  if (end == start) {
    return {Tensor::Shape(ShapeTensor::Tensor({}))};
  }
  if (start == 0 && end == rank) {
    return {Tensor::Shape(shape)};
  }
  ShapeVector slice(end - start);
  for (std::int64_t i = start, j = 0; i < end; ++i, ++j) {
    slice[j] = shape.dims()[i];
  }

  return {Tensor::Shape(ShapeTensor::Tensor(std::move(slice)))};
}

} // namespace vkcnn::details
