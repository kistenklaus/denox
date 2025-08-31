#pragma once

#include "vkcnn/common/model/import/Model_import_AttributeProto.inl"
#include "vkcnn/common/model/import/Model_import_op_Add.inl"
#include "vkcnn/common/model/import/Model_import_op_Conv.inl"
#include "vkcnn/common/model/import/Model_import_op_ReLU.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_node_op(ImportState &state, const onnx::NodeProto &node,
               std::span<const Tensor> inputs,
               const std::unordered_map<std::string, Tensor> &attributes) {
  std::string dom = node.domain();
  // NOTE: Should always exists, otherwise we would have fucked up parsing the
  // top-level opsets.
  opset_version opversion = state.opset_versions.map.at(dom);
  std::string op = node.op_type();

  std::size_t outputCount = static_cast<std::size_t>(node.output_size());

  if (op == "Conv") {
    return import_op_Conv(inputs, attributes, outputCount, opversion);
  } else if (op == "ReLU") {
    return import_op_ReLU(inputs, attributes, outputCount, opversion);
  } else if (op == "Add") {
    return import_op_Add(inputs, attributes, outputCount, opversion);
  }
}

static void import_node(ImportState &state, const onnx::NodeProto &node) {

  std::unordered_map<std::string, Tensor> attributes;
  attributes.reserve(node.attribute_size());

  for (const auto &attrib : node.attribute()) {
    const auto [name, tensor] = parse_attribute(state, attrib, node.name());
    if (attributes.contains(name)) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Node {} has duplicate argument {}", node.name(), name));
    }
    attributes.emplace(name, tensor);
  }
  std::vector<Tensor> inputs;
  inputs.reserve(node.input_size());
  for (const auto &in : node.input()) {
    if (in == "") {
      // optional input
      inputs.push_back(Tensor::Unknown());
      continue;
    }
    auto it = state.tensors.map.find(in);
    if (it == state.tensors.map.end()) {
      throw std::runtime_error(fmt::format(
          "vkcnn: input {} of node {} is undefined.", in, node.name()));
    }
    inputs.push_back(it->second);
  }
  // NOTE: Import op would actually contain the switch over the op type and
  // handle the individual ops.
  auto outputs = import_node_op(state, node, inputs, attributes);
  // NOTE: import_op, should always output the correct number of outputs!
  if (outputs.size() != static_cast<std::size_t>(node.output_size())) {
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} produced the wrong amount of outputs. "
                    "Expected {}, Got {}",
                    node.name(), node.output_size(), outputs.size()));
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const std::string &outputName = node.output(i);
    if (outputName == "") {
      // Optional output, ignore not used anywhere in the graph.
      continue;
    }
    if (outputs[i].isUnknown()) {
      throw std::runtime_error(
          fmt::format("vkcnn: Node {} produces unknown output {}.", node.name(),
                      outputName));
    }
    if (state.tensors.map.contains(outputName)) {
      throw std::runtime_error(
          fmt::format("vkcnn: Node {} produces already existing value ({}). "
                      "Naming collision.",
                      node.name(), outputName));
    }
    state.tensors.map.emplace(outputName, outputs[i]);
  }
}

} // namespace vkcnn::details
