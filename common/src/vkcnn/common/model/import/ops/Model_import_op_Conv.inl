#pragma once

#include "vkcnn/common/model/Model_import_state.inl"
#include "vkcnn/common/model/Model_import_tensors.inl"
#include <span>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Conv(std::span<const Tensor> inputs,
               const std::unordered_map<std::string, Tensor> &attributes,
               std::size_t output_count, opset_version version) {
    
  return {};
}

} // namespace vkcnn::details
