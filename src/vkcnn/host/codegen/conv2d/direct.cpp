#include "./direct.hpp"
#include "vkcnn/host/codegen/entry.hpp"
#include "vkcnn/host/codegen/indent.hpp"
#include "vkcnn/host/codegen/load_tile.hpp"
#include "vkcnn/host/codegen/push_constant.hpp"
#include "vkcnn/host/codegen/shared.hpp"
#include "vkcnn/host/codegen/storage_buffer.hpp"
#include "vkcnn/host/codegen/var.hpp"
#include "vkcnn/host/codegen/version.hpp"
#include <fmt/format.h>
#include <span>

namespace vkcnn::codegen {

void direct_conv2d(const OpConv2d &op, std::string &source) {
  version_preamble("460", source);

  { // Input tensor.
    const Variable inputTensorEntry{Type::F32, "inputTensor", std::dynamic_extent};
    storage_buffer("input_tensor", std::nullopt, 0, 0, Access::ReadOnly,
                   StorageBufferLayout::Std430,
                   std::span<const Variable>(&inputTensorEntry, 1), source);
  }
  { // Output tensor.
    const Variable outputTenorEntry{Type::F32, "outputTensor", std::dynamic_extent};
    storage_buffer("output_tensor", std::nullopt, 0, 1, Access::WriteOnly,
                   StorageBufferLayout::Std430,
                   std::span<const Variable>(&outputTenorEntry, 1), source);
  }
  { // Weight tensor
    const Variable weightTensorEntry{Type::F32, "weightTensor", std::dynamic_extent};
    storage_buffer("weight_tensor", std::nullopt, 0, 2, Access::ReadOnly,
                   StorageBufferLayout::Std430,
                   std::span<const Variable>(&weightTensorEntry, 1), source);
  }
  { // Push constant
    const Variable entries[] = {{Type::U32, "Input_W"}, {Type::U32, "Input_H"}};
    push_constant(entries, "InputExtent", std::nullopt, source);
  }

  define_shared("sh_input", Type::F32,
                (op.tileSize.x + 2 * op.padding.x) *
                    (op.tileSize.y + 2 * op.padding.y) * op.weights.c(),
                source);

  begin_entry(glm::uvec3(op.tileSize.x, op.tileSize.y, 1), "main", source);
  source.append("  const uvec2 groupID = gl_WorkGroupID.xy;\n");
  source.append("  const uvec2 localID = gl_LocalInvocationID.xy;\n");

  load_image_tile_into_shared("inputTensor", op.inputLayout, "sh_input",
                              ImageTensorLayout::HWC, op.tileSize,
                              op.weights.c(), op.padding, op.paddingMode, "localID", "groupID",
                              "Input_W", "Input_H", source);



  source.append(fmt::format("  const ivec2 spos = ivec2(int(localID.x), int(localID.y));\n"));
  source.append(fmt::format("  const uvec2 pos = uvec2(groupID.x * {} + localID.x, groupID.y * {} + localID.y);\n", op.tileSize.x, op.tileSize.y));


  source.append(fmt::format("  for (int k = 0; k < {}; ++k) {{\n", op.weights.c()));


  source.append(fmt::format("    const float v = sh_input[(spos.y + {}) * {} + (spos.x + 1) * {} + k];\n", op.padding.x, (op.tileSize.x + 2 * op.padding.x) * op.weights.c(),
        op.weights.c()));

  source.append(fmt::format("    outputTensor[pos.y * Input_W * {} + pos.x * {} + k] = v;\n", op.weights.c(), op.weights.c()));

  source.append(fmt::format("  }}\n"));


  end_entry(source);
}

} // namespace vkcnn::codegen
