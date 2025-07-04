#pragma once

#include "vkcnn/host/codegen/access.hpp"
#include "vkcnn/host/codegen/type.hpp"
#include "vkcnn/host/codegen/var.hpp"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>
namespace vkcnn::codegen {

enum class StorageBufferLayout {
  Std140,
  Std430
};

void storage_buffer(std::string_view name,
                    std::optional<std::string_view> variableName,
                    unsigned int set, unsigned int binding, Access access,
                    StorageBufferLayout layout,
                    std::span<const Variable> entries, std::string &source);

} // namespace vkcnn::codegen
