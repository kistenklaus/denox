#include "denox/compiler/frontend/onnx/onnx.hpp"
#include "denox/compiler/frontend/model/ModelControlBlock.hpp"
#include "denox/compiler/frontend/onnx/details/ImportState.hpp"
#include "denox/compiler/frontend/onnx/details/import_node.hpp"
#include "denox/compiler/frontend/onnx/details/import_value_info.hpp"
#include "denox/diag/unreachable.hpp"
#include <fmt/format.h>
#include <memory>
#include <onnx.pb.h>

namespace denox::onnx {

static void import_tensor(details::ImportState &state,
                          const ::onnx::TensorProto &tensor) {
  const memory::string &name = tensor.name();
  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "denox: Tensor \"{}\" contains segment, not supported by denox.",
        name));
  }
  if (state.tensors.contains(name)) {
    DENOX_WARN("Tensor \"{}\" is defined multiple times, "
                 "ignoring second occurrence.",
                 name);
    return;
  }
  details::HostTensor h = details::HostTensor::parse(tensor, state.externalDir);
  state.tensors.emplace(name, details::Tensor::Host(std::move(h)));
}

static void import_graph(details::ImportState &state,
                         const ::onnx::GraphProto &graph,
                         const compiler::CompileOptions &options) {
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

  for (const auto &input : runtime_inputs) {
    // We probably need a special function for input / output
    import_value_info(state, *input, details::ValueInfoImportContext::Input,
                      options);
  }

  for (const ::onnx::NodeProto &node : graph.node()) {
    details::import_node(state, node);
  }
  for (const auto &value_info : graph.value_info()) {
    import_value_info(state, value_info, details::ValueInfoImportContext::Hint,
                      options);
  }
  for (const auto &output : graph.output()) {
    import_value_info(state, output, details::ValueInfoImportContext::Output,
                      options);
  }
}

compiler::Model read(memory::span<const std::byte> raw,
                     const compiler::CompileOptions &options) {
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
    controlBlock->meta.domain =
        onnx.domain().empty() ? memory::nullopt
                              : memory::optional<memory::string>(onnx.domain());
    controlBlock->meta.producerName =
        onnx.producer_name().empty()
            ? memory::nullopt
            : memory::optional<memory::string>(onnx.producer_name());
    controlBlock->meta.producerVersion =
        onnx.producer_version().empty()
            ? memory::nullopt
            : memory::optional<memory::string>(onnx.producer_version());
    controlBlock->meta.modelVersion =
        onnx.model_version() == 0
            ? memory::nullopt
            : memory::optional<memory::string>(
                  fmt::format("{}", onnx.model_version()));

    details::ImportState state{
        .externalDir = denox::io::Path::cwd(),
        .symGraph = &controlBlock->symGraph,
        .output = compiler::Model(std::move(controlBlock)),
        .ir_version = onnx.ir_version(),
        .producer_name = onnx.producer_name(),
        .producer_version = onnx.producer_version(),
        .domain = onnx.domain(),
        .model_version = onnx.model_version(),
        .opset_versions = {},
        .tensors = {},
    };

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

    import_graph(state, onnx.graph(), options);

    return std::move(state.output);
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(
        fmt::format("vkcnn: Failed to import ONNX model: {}", e.what()));
  }
  diag::unreachable();
}

} // namespace denox::onnx
