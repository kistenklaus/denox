#pragma once

#include <fmt/format.h>
#include <string>
#include <string_view>
namespace vkcnn::codegen {

inline void version_preamble(std::string_view version, std::string &src) {
  src.append(fmt::format("#version {}\n", version));
  src.append("#extension GL_KHR_shader_subgroup_basic : enable\n");

}

} // namespace vkcnn::codegen
