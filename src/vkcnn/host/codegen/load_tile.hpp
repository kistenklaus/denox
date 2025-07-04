#pragma once

#include "vkcnn/ImageTensor.hpp"
#include "vkcnn/host/ImageTensorLayout.hpp"
#include "vkcnn/host/ops/PaddingMode.hpp"
#include <cassert>
#include <glm/ext/vector_uint2.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
namespace vkcnn::codegen {

void load_image_tile_into_shared(std::string_view globalVar,
                                 ImageTensorLayout globalLayout,
                                 std::string_view sharedVar,
                                 ImageTensorLayout sharedLayout,
                                 glm::uvec2 tileSize, 
                                 unsigned int channels,
                                 glm::uvec2 padding,
                                 PaddingMode paddingMode,
                                 std::optional<std::string_view> localId,
                                 std::optional<std::string_view> groupId,
                                 std::string_view inputW,
                                 std::string_view inputH,
                                 std::string &source, int indentation = 2);

} // namespace vkcnn::codegen
