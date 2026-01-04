#pragma once

#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/compiler/frontend/onnx/details/OpSetVersions.hpp"
#include "denox/compiler/frontend/onnx/details/values/Tensor.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/symbolic/Sym.hpp"
#include <string>

namespace denox::onnx::details {

struct SymbolName {
  Sym sym;
  std::string name;
};

struct ImportState {
  denox::io::Path externalDir;
  SymGraph *symGraph;
  compiler::Model output;

  std::int64_t ir_version;
  std::string producer_name;
  denox::memory::string producer_version;
  denox::memory::string domain;
  std::int64_t model_version;

  OpSetVersions opset_versions;

  memory::hash_map<memory::string, Tensor> tensors;
};

} // namespace denox::onnx::details
