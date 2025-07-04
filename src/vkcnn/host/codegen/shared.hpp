#pragma once

#include "vkcnn/host/codegen/type.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <optional>
#include <string_view>

namespace vkcnn::codegen {

void define_shared(std::string_view var, Type type,
                   std::optional<unsigned int> arraySize,
                   std::string& source);

} // namespace vkcnn::codegen
