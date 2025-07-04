#pragma once

#include <glm/ext/vector_uint3.hpp>
#include <string>
namespace vkcnn::codegen {

void begin_entry(glm::uvec3 workgroupSize, std::string_view entry,
                 std::string &source);

void end_entry(std::string &source);

}
