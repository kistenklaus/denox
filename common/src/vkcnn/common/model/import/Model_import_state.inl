#pragma once

#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include "vkcnn/common/model/SymTensorExtent.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
namespace vkcnn::details {

using opset_version = std::int64_t;
struct OpSetVersions {
  opset_version core_version;

  opset_version operator[](const std::string &domain) {
    assert(map.contains(domain));
    return map[domain];
  }

  std::unordered_map<std::string, opset_version> map;
};

// -----------------------------------------------------------------------------
// One global table (name -> Tensor). Constants are those with kind()==Constant.
// Shape/size/index inputs are UInt meta tensors (kind()==UInt).
struct TensorMap {
  std::unordered_map<std::string, Tensor> map;

  bool has(std::string_view name) const {
    return map.find(std::string(name)) != map.end();
  }
  const Tensor &at(std::string_view name) const {
    auto it = map.find(std::string(name));
    if (it == map.end())
      throw std::out_of_range("TensorMap: missing tensor");
    return it->second;
  }

  Tensor &at(std::string_view name) {
    auto it = map.find(std::string(name));
    if (it == map.end())
      throw std::out_of_range("TensorMap: missing tensor");
    return it->second;
  }
};

using SymMap = std::unordered_map<std::string, Sym>;

struct ImportState {

  std::string model_dir;

  std::int64_t ir_version;
  std::string producer_name;
  std::string producer_version;
  std::string domain;
  std::int64_t model_version;
  OpSetVersions opset_versions;

  TensorMap tensors;

  SymMap symbolMap;
  SymTensorExtent inputExtent;

  std::shared_ptr<vkcnn::SymGraph> symGraph;
  vkcnn::Model output;
};

} // namespace vkcnn::details
