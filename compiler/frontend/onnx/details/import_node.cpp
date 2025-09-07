#include "frontend/onnx/details/import_node.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/details/ops/ops.hpp"
#include "frontend/onnx/details/values/Attribute.hpp"
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

  if (op == "Abs") {
    return ops::abs(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Add") {
    return ops::add(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "AveragePool") {
    return ops::average_pool(state, inputs, outputCount, attributes, opversion,
                             node.name());
  } else if (op == "Cast") {
    return ops::cast(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Ceil") {
    return ops::ceil(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Clip") {
    return ops::ceil(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Concat") {
    return ops::concat(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Constant") {
    return ops::constant(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "ConstantOfShape") {
    return ops::constant_of_shape(state, inputs, outputCount, attributes,
                                  opversion, node.name());
  } else if (op == "Conv") {
    return ops::conv(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Div") {
    return ops::div(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Expand") {
    return ops::expand(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Floor") {
    return ops::floor(state, inputs, outputCount, attributes, opversion,
                      node.name());
  } else if (op == "Gather") {
    return ops::gather(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "LeakyRelu") {
    return ops::leaky_relu(state, inputs, outputCount, attributes, opversion,
                           node.name());
  } else if (op == "Max") {
    return ops::max(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "MaxPool") {
    return ops::max_pool(state, inputs, outputCount, attributes, opversion,
                         node.name());
  } else if (op == "Min") {
    return ops::min(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Mod") {
    return ops::mod(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Mul") {
    return ops::mul(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Neg") {
    return ops::neg(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Pad") {
    return ops::pad(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Pow") {
    return ops::pow(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Reciprocal") {
    return ops::reciprocal(state, inputs, outputCount, attributes, opversion,
                           node.name());
  } else if (op == "Relu") {
    return ops::relu(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Reshape") {
    return ops::reshape(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Resize") {
    return ops::resize(state, inputs, outputCount, attributes, opversion,
                       node.name());
  } else if (op == "Round") {
    return ops::round(state, inputs, outputCount, attributes, opversion,
                      node.name());
  } else if (op == "Shape") {
    return ops::shape(state, inputs, outputCount, attributes, opversion,
                      node.name());
  } else if (op == "Sign") {
    return ops::sign(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Slice") {
    return ops::slice(state, inputs, outputCount, attributes, opversion,
                      node.name());
  } else if (op == "Sqrt") {
    return ops::sqrt(state, inputs, outputCount, attributes, opversion,
                     node.name());
  } else if (op == "Squeeze") {
    return ops::squeeze(state, inputs, outputCount, attributes, opversion,
                        node.name());
  } else if (op == "Sub") {
    return ops::sub(state, inputs, outputCount, attributes, opversion,
                    node.name());
  } else if (op == "Transpose") {
    return ops::transpose(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else if (op == "Unsqueeze") {
    return ops::unsqueeze(state, inputs, outputCount, attributes, opversion,
                          node.name());
  } else {
    throw std::runtime_error(
        fmt::format("vkcnn: operation {} is not supported (node = \"{}\")", op,
                    node.name()));
  }
  denox::compiler::diag::unreachable();
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
