#pragma once

#include "vkcnn/common/model/Model_import_state.inl"
#include "vkcnn/common/model/Model_import_tensors.inl"
#include <vector>
namespace vkcnn::details {


static std::vector<Tensor>
import_op_Add(std::span<const Tensor> inputs,
               const std::unordered_map<std::string, Tensor> &attributes,
               std::size_t output_count, opset_version version) {
  return {};
}


}
