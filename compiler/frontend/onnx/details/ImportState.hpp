#pragma once

#include "frontend/onnx/details/OpSetVersions.hpp"
#include "frontend/onnx/details/values/Tensor.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/hashmap.hpp"
#include "model/Model.hpp"
#include "symbolic/Sym.hpp"
#include <string>

namespace denox::onnx::details {

struct ImportState {
  denox::io::Path externalDir;
  compiler::SymGraph* symGraph;
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
