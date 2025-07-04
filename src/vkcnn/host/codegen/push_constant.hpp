
#pragma once

#include "vkcnn/host/codegen/var.hpp"
#include <optional>
#include <span>
#include <string>
namespace vkcnn::codegen {

void push_constant(std::span<const Variable> entries, std::string_view name,
                   std::optional<std::string_view> var, std::string &source);

} // namespace vkcnn::codegen
