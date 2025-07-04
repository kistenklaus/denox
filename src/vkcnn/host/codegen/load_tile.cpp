#include "./load_tile.hpp"
#include "vkcnn/host/codegen/indent.hpp"
#include "vkcnn/host/ops/PaddingMode.hpp"
#include <chrono>
#include <fmt/base.h>
#include <print>

static void load_image_tile_into_shared_from_HWC_to_HWC(
    std::string_view globalVar, std::string_view sharedVar, glm::uvec2 tileSize,
    unsigned int channels, glm::uvec2 padding, vkcnn::PaddingMode paddingMode,
    std::optional<std::string_view> localIdOpt,
    std::optional<std::string_view> groupIdOpt, std::string_view inputW,
    std::string_view inputH, std::string &source, int ind) {

  std::string_view localID = localIdOpt.value_or("gl_LocalInvocationID");
  std::string_view groupID = groupIdOpt.value_or("gl_WorkGroupID");

  using namespace vkcnn::codegen;
  // begin code block
  indent(ind, source);
  source.append("{\n");
  ind += 2;

  // actual logic

  unsigned int sharedH = tileSize.x + 2 * padding.x;
  unsigned int sharedW = tileSize.y + 2 * padding.y;

  const unsigned int SUBGROUP_SIZE = 32;

  const unsigned int sharedRowSize = sharedW * channels;

  const unsigned int workgroupSize = tileSize.x * tileSize.y;
  const unsigned int subgroupCount =
      (workgroupSize + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
  unsigned int steps = (sharedH + subgroupCount - 1) / subgroupCount;
  unsigned int substeps = (sharedRowSize + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

  indent(ind, source);
  source.append(
      fmt::format("const int rowSize = int({}) * {};\n", inputW, channels));
  indent(ind, source);
  source.append(fmt::format(
      "const ivec2 tile = ivec2(int({}.x * {}) - {}, int({}.y * {}) - {});\n",
      groupID, tileSize.x, padding.x, groupID, tileSize.y, padding.y));

  indent(ind, source);
  source.append(fmt::format(
      "const int tileStart = tile.y * int(rowSize) + tile.x * {};\n",
      channels));

  for (unsigned int i = 0; i < steps; ++i) {
    if (i == 0) {
      indent(ind, source);
      source.append(fmt::format("const int h{} = int(gl_SubgroupID);\n", i));
    } else {
      indent(ind, source);
      source.append(fmt::format("const int h{} = int(gl_SubgroupID) + {};\n", i,
                                i * subgroupCount));
    }

    indent(ind, source);
    source.append(
        fmt::format("const int hs{} = h{} * {};\n", i, i, sharedRowSize));

    indent(ind, source);
    source.append(fmt::format("const int hg{} = h{} * rowSize;\n", i, i));

    bool enableTopPadding = i == 0;
    bool enableBottomPadding = i == steps - 1;
    assert(!(enableTopPadding && enableBottomPadding));

    unsigned int remainingRows = sharedH - i * subgroupCount;
    if (remainingRows < subgroupCount) {
      indent(ind, source);
      source.append(fmt::format("if (gl_SubgroupID < {}) {{\n", remainingRows));
      ind += 2;
    }

    if (enableTopPadding && enableBottomPadding) {
      throw std::runtime_error("NOT-YET-IMPLEMENTED");
    } else if (enableTopPadding) {
      indent(ind, source);
      source.append(fmt::format("if (tile.y + h{} < 0) {{\n", i, padding.y));

      ind += 2;
      for (uint ix = 0; ix < substeps; ++ix) {
        switch (paddingMode) {
        case vkcnn::PaddingMode::ZERO:
          indent(ind, source);
          source.append(
              fmt::format("{}[hs{} + gl_SubgroupInvocationID + {}] = 0.0;\n",
                          sharedVar, i, ix * SUBGROUP_SIZE));
          break;
        }
      }

      ind -= 2;
      indent(ind, source);
      source.append(fmt::format("}} else {{\n"));
      ind += 2;

    } else if (enableBottomPadding) {

      indent(ind, source);
      source.append(fmt::format("if (tile.y + h{} >= {}) {{\n", i, inputH));
      ind += 2;

      for (uint ix = 0; ix < substeps; ++ix) {
        switch (paddingMode) {
        case vkcnn::PaddingMode::ZERO:
          indent(ind, source);
          source.append(
              fmt::format("{}[hs{} + gl_SubgroupInvocationID + {}] = 0.0;\n",
                          sharedVar, i, ix * SUBGROUP_SIZE));
          break;
        }
      }

      ind -= 2;
      indent(ind, source);
      source.append(fmt::format("}} else {{\n"));
      ind += 2;
    }

    for (uint ix = 0; ix < substeps; ++ix) {
      uint remaining = sharedRowSize - ix * SUBGROUP_SIZE;
      if (remaining < SUBGROUP_SIZE) {
        // check for overhead.
        indent(ind, source);
        source.append(
            fmt::format("if (gl_SubgroupInvocationID < {}) {{\n", remaining));
        ind += 2;
      }

      bool frontEdge = ix == 0;
      bool backEdge = ix == substeps - 1;
      assert(!(frontEdge && backEdge));

      if (frontEdge) {
        assert(ix == 0);
        indent(ind, source);
        source.append(
            fmt::format("if ({} * tile.x + int(gl_SubgroupInvocationID) < 0) {{\n",
                        channels, remaining));
        ind += 2;
        switch (paddingMode) {
        case vkcnn::PaddingMode::ZERO:
          indent(ind, source);
          source.append(
              fmt::format("{}[hs{} + gl_SubgroupInvocationID + {}] = 0.0;\n",
                          sharedVar, i, ix * SUBGROUP_SIZE));
          break;
        }
        ind -= 2;
        indent(ind, source);
        source.append("} else {\n");

        ind += 2;
      }
      if (backEdge) {
        indent(ind, source);
        source.append(fmt::format(
            "if ({} * tile.x + int(gl_SubgroupInvocationID) + {} >= {} * {}) {{\n",
            channels, ix * SUBGROUP_SIZE, inputW, channels));
        ind += 2;
        switch (paddingMode) {
        case vkcnn::PaddingMode::ZERO:
          indent(ind, source);
          source.append(
              fmt::format("{}[hs{} + gl_SubgroupInvocationID + {}] = 0.0;\n",
                          sharedVar, i, ix * SUBGROUP_SIZE));
          break;
        }
        ind -= 2;
        indent(ind, source);
        source.append("} else {\n");

        ind += 2;
      }

      indent(ind, source);
      source.append(fmt::format(
          "{}[hs{} + gl_SubgroupInvocationID + {}] = {}[tileStart + hg{} + "
          "gl_SubgroupInvocationID + {}];\n",
          sharedVar, i, ix * SUBGROUP_SIZE, globalVar, i, ix * SUBGROUP_SIZE));

      if (frontEdge || backEdge) {
        ind -= 2;
        indent(ind, source);
        source.append("}\n");
      }

      if (remaining < SUBGROUP_SIZE) {
        ind -= 2;
        indent(ind, source);
        source.append("}\n");
      }
    }

    //// Handle top padding.
    if (enableTopPadding || enableBottomPadding) {
      ind -= 2;
      indent(ind, source);
      source.append(fmt::format("}}\n"));
    }

    if (remainingRows < subgroupCount) {
      ind -= 2;
      indent(ind, source);
      source.append("}\n");
    }

    indent(ind, source);
    source.append("\n");
  }

  indent(ind, source);
  source.append("memoryBarrierShared();\n");

  indent(ind, source);
  source.append("barrier();\n");

  // end code block
  ind -= 2;
  indent(ind, source);
  source.append("}\n");
}

void vkcnn::codegen::load_image_tile_into_shared(
    std::string_view globalVar, ImageTensorLayout globalLayout,
    std::string_view sharedVar, ImageTensorLayout sharedLayout,
    glm::uvec2 tileSize, unsigned int channels, glm::uvec2 padding,
    PaddingMode paddingMode, std::optional<std::string_view> localId,
    std::optional<std::string_view> groupId, std::string_view inputW,
    std::string_view inputH, std::string &source, int indentation) {
  switch (globalLayout) {
  case ImageTensorLayout::HWC:
    switch (sharedLayout) {
    case ImageTensorLayout::HWC:
      load_image_tile_into_shared_from_HWC_to_HWC(
          globalVar, sharedVar, tileSize, channels, padding, paddingMode,
          localId, groupId, inputW, inputH, source, indentation);
      break;
    case ImageTensorLayout::CHW:
      throw std::runtime_error("NOT-IMPLEMENTED-YET");
    }
    break;
  case ImageTensorLayout::CHW:
    throw std::runtime_error("NOT-IMPLEMENTED-YET");
  }
}
