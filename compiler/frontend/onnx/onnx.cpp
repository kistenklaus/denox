#include "frontend/onnx/onnx.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/details/ImportState.hpp"
#include "frontend/onnx/details/import_node.hpp"
#include "frontend/onnx/details/import_value_info.hpp"
#include "model/ModelControlBlock.hpp"
#include <memory>
#include <onnx.pb.h>

namespace denox::onnx {

static void import_tensor(details::ImportState &state,
                          const ::onnx::TensorProto &tensor) {
  const memory::string &name = tensor.name();
  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        name));
  }
  if (state.tensors.contains(name)) {
    fmt::println("vkcnn: [Warning]: Tensor \"{}\" is defined multiple times, "
                 "ignoring second occurrence.",
                 name);
    return;
  }
  details::HostTensor h = details::HostTensor::parse(tensor, state.externalDir);
  state.tensors.emplace(name, details::Tensor::Host(std::move(h)));
}

static void import_graph(details::ImportState &state,
                         const ::onnx::GraphProto &graph) {
  if (graph.sparse_initializer_size() != 0) {
    throw std::runtime_error("vkcnn: Model contains sparse initializers are "
                             "not supported by vkcnn.");
  }

  for (const auto &tensor : graph.initializer()) {
    import_tensor(state, tensor);
  }

  memory::vector<const ::onnx::ValueInfoProto *> runtime_inputs;
  for (const auto &in : graph.input()) {
    if (!state.tensors.contains(in.name())) {
      runtime_inputs.push_back(&in);
    }
  }
  if (runtime_inputs.size() != 1) {
    throw std::runtime_error(
        fmt::format("vkcnn: Expected exactly one runtime input (image). Got {}",
                    runtime_inputs.size()));
  }

  if (graph.output().size() != 1) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Expected exactly one output. Got {}", graph.output_size()));
  }

  const auto &input = *runtime_inputs.front();
  const auto &output = graph.output(0);

  // We probably need a special function for input / output
  details::import_value_info(state, input,
                             details::ValueInfoImportContext::Input);

  for (const ::onnx::NodeProto &node : graph.node()) {
    details::import_node(state, node);
  }
  for (const auto &value_info : graph.value_info()) {
    import_value_info(state, value_info, details::ValueInfoImportContext::Hint);
  }
  import_value_info(state, output, details::ValueInfoImportContext::Output);
}

compiler::Model read(memory::span<const std::byte> raw, io::Path onnx_dir) {
  try {
    ::onnx::ModelProto onnx;
    if (!onnx.ParseFromArray(raw.data(), static_cast<int>(raw.size_bytes()))) {
      throw std::runtime_error("Failed to parse ONNX protobuf");
    }
    if (onnx.functions_size() != 0) {
      throw std::runtime_error("vkcnn: ONNX functions are not supported.");
    }
    if (onnx.opset_import_size() == 0) {
      throw std::runtime_error("vkcnn: missing opset_import.");
    }
    if (!onnx.has_graph()) {
      throw std::runtime_error("vkcnn: missing top-level graph.");
    }

    auto controlBlock =
        std::make_unique<denox::compiler::details::model::ModelControlBlock>();

    details::ImportState state{.externalDir = onnx_dir,
                               .symGraph = &controlBlock->symGraph,
                               .output =
                                   compiler::Model(std::move(controlBlock)),
                               .ir_version = onnx.ir_version(),
                               .producer_name = onnx.producer_name(),
                               .producer_version = onnx.producer_version(),
                               .domain = onnx.domain(),
                               .model_version = onnx.model_version(),
                               .opset_versions = {},
                               .tensors = {}};

    state.opset_versions.map.clear();
    for (int i = 0; i < onnx.opset_import_size(); ++i) {
      const auto &imp = onnx.opset_import(i);
      const memory::string dom =
          imp.domain().empty() ? "ai.onnx" : imp.domain();
      const opset_version ver = imp.version();

      auto it = state.opset_versions.map.find(dom);
      if (it == state.opset_versions.map.end() || it->second < ver) {
        state.opset_versions.map[dom] = ver;
      }
    }

    auto core_it = state.opset_versions.map.find("ai.onnx");
    if (core_it == state.opset_versions.map.end() || core_it->second <= 0) {
      throw std::runtime_error(
          "vkcnn: missing or invalid core opset (ai.onnx).");
    }
    state.opset_versions.core_version = core_it->second;
    state.opset_versions.map.emplace("", core_it->second);

    for (const auto &kv : state.opset_versions.map) {
      const memory::string &dom = kv.first;
      const opset_version ver = kv.second;
      if (dom != "ai.onnx" && dom != "") {
        throw std::runtime_error("vkcnn: unsupported operator set domain \"" +
                                 dom + "\" (version " + std::to_string(ver) +
                                 ")");
      }
    }

    import_graph(state, onnx.graph());

    return std::move(state.output);
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Failed to import ONNX model: {}\n\n{}\n",
        onnx_dir.empty() ? memory::string("<unknown>") : onnx_dir.str(),
        e.what()));
  }
  denox::compiler::diag::unreachable();
}

} // namespace denox::onnx
