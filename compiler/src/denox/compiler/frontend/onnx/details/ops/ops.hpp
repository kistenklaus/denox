#pragma once

#include "denox/compiler/frontend/onnx/details/ImportState.hpp"
#include "denox/compiler/frontend/onnx/details/values/Attribute.hpp"
#include "denox/compiler/frontend/onnx/details/values/Tensor.hpp"

namespace denox::onnx::details::ops {

memory::vector<Tensor>
abs(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
add(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
average_pool(ImportState &state,
             memory::span<const memory::optional<Tensor>> inputs,
             std::size_t outputCount,
             const memory::hash_map<memory::string, Attribute> &attributes,
             opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
cast(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
ceil(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
clip(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
concat(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
       std::size_t outputCount,
       const memory::hash_map<memory::string, Attribute> &attributes,
       opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
constant(ImportState &state,
         memory::span<const memory::optional<Tensor>> inputs,
         std::size_t outputCount,
         const memory::hash_map<memory::string, Attribute> &attributes,
         opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
constant_of_shape(ImportState &state,
                  memory::span<const memory::optional<Tensor>> inputs,
                  std::size_t outputCount,
                  const memory::hash_map<memory::string, Attribute> &attributes,
                  opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
conv(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
div(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
expand(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
       std::size_t outputCount,
       const memory::hash_map<memory::string, Attribute> &attributes,
       opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
floor(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
      std::size_t outputCount,
      const memory::hash_map<memory::string, Attribute> &attributes,
      opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
gather(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
       std::size_t outputCount,
       const memory::hash_map<memory::string, Attribute> &attributes,
       opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
leaky_relu(ImportState &state,
           memory::span<const memory::optional<Tensor>> inputs,
           std::size_t outputCount,
           const memory::hash_map<memory::string, Attribute> &attributes,
           opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
max(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
max_pool(ImportState &state,
         memory::span<const memory::optional<Tensor>> inputs,
         std::size_t outputCount,
         const memory::hash_map<memory::string, Attribute> &attributes,
         opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
min(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
mod(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
mul(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor> neg(
    ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> & attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
pad(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
pow(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
reciprocal(ImportState &state,
           memory::span<const memory::optional<Tensor>> inputs,
           std::size_t outputCount,
           const memory::hash_map<memory::string, Attribute> &attributes,
           opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
relu(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
reshape(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
        std::size_t outputCount,
        const memory::hash_map<memory::string, Attribute> &attributes,
        opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
resize(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
       std::size_t outputCount,
       const memory::hash_map<memory::string, Attribute> &attributes,
       opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
round(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
      std::size_t outputCount,
      const memory::hash_map<memory::string, Attribute> &attributes,
      opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
shape(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
      std::size_t outputCount,
      const memory::hash_map<memory::string, Attribute> &attributes,
      opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
sign(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
slice(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
      std::size_t outputCount,
      const memory::hash_map<memory::string, Attribute> &attributes,
      opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
sqrt(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
     std::size_t outputCount,
     const memory::hash_map<memory::string, Attribute> &attributes,
     opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
squeeze(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
        std::size_t outputCount,
        const memory::hash_map<memory::string, Attribute> &attributes,
        opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
sub(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
transpose(ImportState &state,
          memory::span<const memory::optional<Tensor>> inputs,
          std::size_t outputCount,
          const memory::hash_map<memory::string, Attribute> &attributes,
          opset_version version, memory::string_view nodeName);

memory::vector<Tensor>
unsqueeze(ImportState &state,
          memory::span<const memory::optional<Tensor>> inputs,
          std::size_t outputCount,
          const memory::hash_map<memory::string, Attribute> &attributes,
          opset_version version, memory::string_view nodeName);

} // namespace denox::onnx::details::ops
