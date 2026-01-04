#include "denox/compiler/frontend/onnx/details/import_node.hpp"
#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"
#include "denox/compiler/frontend/onnx/details/values/Attribute.hpp"
#include <onnx.pb.h>

namespace denox::onnx::details {

static std::vector<Tensor>
import_node_op(ImportState &state, const ::onnx::NodeProto &node,
               std::span<const std::optional<Tensor>> inputs,
               const std::unordered_map<std::string, Attribute> &attributes) {

  std::string dom = node.domain();
  opset_version opversion = state.opset_versions.map.at(dom);
  std::string op = node.op_type();

  std::size_t outputCount = static_cast<std::size_t>(node.output_size());

  memory::vector<Tensor> outputs;
  if (op == "Abs") {
    outputs = ops::abs(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Add") {
    outputs = ops::add(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "AveragePool") {
    outputs = ops::average_pool(state, inputs, outputCount, attributes,
                                opversion, node.name());
  } else if (op == "Cast") {
    outputs = ops::cast(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Ceil") {
    outputs = ops::ceil(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Clip") {
    outputs = ops::ceil(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Concat") {
    outputs = ops::concat(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else if (op == "Constant") {
    outputs = ops::constant(state, inputs, outputCount, attributes, opversion,
                            node.name());
  } else if (op == "ConstantOfShape") {
    outputs = ops::constant_of_shape(state, inputs, outputCount, attributes,
                                     opversion, node.name());
  } else if (op == "Conv") {
    outputs = ops::conv(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Div") {
    outputs = ops::div(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Expand") {
    outputs = ops::expand(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else if (op == "Floor") {
    outputs = ops::floor(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "Gather") {
    outputs = ops::gather(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else if (op == "LeakyRelu") {
    outputs = ops::leaky_relu(state, inputs, outputCount, attributes, opversion,
                              node.name());
  } else if (op == "Max") {
    outputs = ops::max(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "MaxPool") {
    outputs = ops::max_pool(state, inputs, outputCount, attributes, opversion,
                            node.name());
  } else if (op == "Min") {
    outputs = ops::min(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Mod") {
    outputs = ops::mod(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Mul") {
    outputs = ops::mul(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Neg") {
    outputs = ops::neg(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Pad") {
    outputs = ops::pad(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Pow") {
    outputs = ops::pow(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Reciprocal") {
    outputs = ops::reciprocal(state, inputs, outputCount, attributes, opversion,
                              node.name());
  } else if (op == "Relu") {
    outputs = ops::relu(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Reshape") {
    outputs = ops::reshape(state, inputs, outputCount, attributes, opversion,
                           node.name());
  } else if (op == "Resize") {
    outputs = ops::resize(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else if (op == "Round") {
    outputs = ops::round(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "Shape") {
    outputs = ops::shape(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "Sign") {
    outputs = ops::sign(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Slice") {
    outputs = ops::slice(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "Sqrt") {
    outputs = ops::sqrt(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Squeeze") {
    outputs = ops::squeeze(state, inputs, outputCount, attributes, opversion,
                           node.name());
  } else if (op == "Sub") {
    outputs = ops::sub(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Transpose") {
    outputs = ops::transpose(state, inputs, outputCount, attributes, opversion,
                             node.name());
  } else if (op == "Unsqueeze") {
    outputs = ops::unsqueeze(state, inputs, outputCount, attributes, opversion,
                             node.name());
  } else {
    throw std::runtime_error(
        fmt::format("vkcnn: operation {} is not supported (node = \"{}\")", op,
                    node.name()));
  }
  assert(outputs.size() == outputCount);
  for (size_t o = 0; o < outputs.size(); ++o) {
    if (outputs[o].isDevice()) {
      outputs[o].device().handle().setName(node.output(static_cast<int>(o)));
    }
  }

  return outputs;
}

void import_node(ImportState &state, const ::onnx::NodeProto &node) {
  std::unordered_map<std::string, Attribute> attributes;
  attributes.reserve(static_cast<std::size_t>(node.attribute_size()));
  for (const auto &attrib : node.attribute()) {
    const NamedAttribute attribute =
        Attribute::parse(attrib, state.externalDir, node.name());
    if (attributes.contains(attribute.name)) {
      throw std::runtime_error(
          fmt::format("vkcnn: Node {} has duplicate attribute {}", node.name(),
                      attribute.name));
    }
    attributes.emplace(attribute.name, attribute.attribute);
  }
  std::vector<std::optional<Tensor>> inputs;
  inputs.reserve(static_cast<std::size_t>(node.input_size()));
  for (const auto &in : node.input()) {
    if (in == "") {
      inputs.push_back(std::nullopt);
      continue;
    }
    auto it = state.tensors.find(in);
    if (it == state.tensors.end()) {
      throw std::runtime_error(fmt::format(
          "vkcnn: input {} of node {} is undefined.", in, node.name()));
    }
    inputs.push_back(it->second);
  }
  auto outputs = import_node_op(state, node, inputs, attributes);
  if (outputs.size() != static_cast<std::size_t>(node.output_size())) {
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} produced the wrong amount of outputs. "
                    "Expected {}, Got {}",
                    node.name(), node.output_size(), outputs.size()));
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const std::string &outputName = node.output(static_cast<int>(i));
    if (outputName == "") {
      continue;
    }

    if (state.tensors.contains(outputName)) {
      throw std::runtime_error(
          fmt::format("vkcnn: Node {} produces already existing value ({}). "
                      "Naming collision.",
                      node.name(), outputName));
    }
    state.tensors.emplace(outputName, outputs[i]);
  }
}

} // namespace denox::onnx::details
