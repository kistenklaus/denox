#include "./Model.hpp"
#include <filesystem>
#include <fmt/base.h>
#include <fstream>
#include <memory>
#include <onnx.pb.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vkcnn/common/model/import/Model_import_NodeProto.inl"
#include "vkcnn/common/model/import/Model_import_TensorProto.inl"
#include "vkcnn/common/model/import/Model_import_ValueInfoProto.inl"
#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"

namespace vkcnn {

namespace details {

static void import_graph(ImportState &state, const onnx::GraphProto &graph) {
  if (graph.sparse_initializer_size() != 0) {
    throw std::runtime_error("vkcnn: Model contains sparse initializers are "
                             "not supported by vkcnn.");
  }

  for (const auto &tensor : graph.initializer()) {
    import_tensor(state, tensor);
  }

  std::vector<const onnx::ValueInfoProto *> runtime_inputs;
  for (const auto &in : graph.input()) {
    if (!state.tensors.has(in.name())) {
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
  import_value_info(state, input, ValueInfoImportContext::Input);

  for (const onnx::NodeProto &node : graph.node()) {
    import_node(state, node);
  }
  for (const auto &value_info : graph.value_info()) {
    import_value_info(state, value_info, ValueInfoImportContext::Hint);
  }
  import_value_info(state, output, ValueInfoImportContext::Output);
}

static void import_model(ImportState &state, const onnx::ModelProto &model) {
  state.ir_version = model.ir_version();
  state.producer_name = model.producer_name();
  state.producer_version = model.producer_version();
  state.domain = model.domain(); // informational
  state.model_version = model.model_version();

  if (model.functions_size() != 0) {
    throw std::runtime_error("vkcnn: ONNX functions are not supported.");
  }
  if (model.opset_import_size() == 0) {
    throw std::runtime_error("vkcnn: missing opset_import.");
  }
  if (!model.has_graph()) {
    throw std::runtime_error("vkcnn: missing top-level graph.");
  }

  state.opset_versions.map.clear();
  for (int i = 0; i < model.opset_import_size(); ++i) {
    const auto &imp = model.opset_import(i);
    const std::string dom = imp.domain().empty() ? "ai.onnx" : imp.domain();
    const opset_version ver = static_cast<opset_version>(imp.version());

    auto it = state.opset_versions.map.find(dom);
    if (it == state.opset_versions.map.end() || it->second < ver) {
      state.opset_versions.map[dom] = ver;
    }
  }

  auto core_it = state.opset_versions.map.find("ai.onnx");
  if (core_it == state.opset_versions.map.end() || core_it->second <= 0) {
    throw std::runtime_error("vkcnn: missing or invalid core opset (ai.onnx).");
  }
  state.opset_versions.core_version = core_it->second;
  state.opset_versions.map.emplace("", core_it->second);

  for (const auto &kv : state.opset_versions.map) {
    const std::string &dom = kv.first;
    const opset_version ver = kv.second;
    if (dom != "ai.onnx" && dom != "") {
      throw std::runtime_error("vkcnn: unsupported operator set domain \"" +
                               dom + "\" (version " + std::to_string(ver) +
                               ")");
    }
  }

  import_graph(state, model.graph());
}

} // namespace details

Model Model::import(std::string_view path_str) {
  details::ImportState state;

  const std::string path(path_str);
  state.model_dir = std::filesystem::path(path).parent_path();
  state.symGraph = state.output.m_controlBlock->symGraph;

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("vkcnn: cannot open ONNX file: " + path);
  ::onnx::ModelProto onnx;
  if (!onnx.ParseFromIstream(&ifs)) {
    throw std::runtime_error("vkcnn: Failed to parse ONNX protobuf");
  }

  import_model(state, onnx);
  return state.output;
}

} // namespace vkcnn
