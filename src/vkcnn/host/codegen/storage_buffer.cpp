#include "./storage_buffer.hpp"
#include <span>

namespace vkcnn::codegen {

void storage_buffer(std::string_view name,
                    std::optional<std::string_view> variableName,
                    unsigned int set, unsigned int binding, Access access,
                    StorageBufferLayout layout,
                    std::span<const Variable> entries, std::string &source) {
  source.append(fmt::format("layout(set = {}, binding = {}, ", set, binding));
  switch (layout) {
  case StorageBufferLayout::Std140:
    source.append("std140");
    break;
  case StorageBufferLayout::Std430:
    source.append("std430");
    break;
  }
  source.append(") ");

  switch (access) {
  case Access::ReadWrite:
    break;
  case Access::ReadOnly:
    source.append("readonly ");
    break;
  case Access::WriteOnly:
    source.append("writeonly ");
    break;
  }
  source.append("buffer ");
  source.append(name);
  source.append(" {\n");
  for (auto var : entries) {
    if (var.arraySize.has_value()) {
      if (var.arraySize.value() == std::dynamic_extent) {
        source.append(fmt::format("  {} {}[];\n", var.type, var.name));
      } else {
        source.append(fmt::format("  {} {}[{}];\n", var.type, var.name, var.arraySize.value()));
      }
    } else {
      source.append(fmt::format("  {} {};\n", var.type, var.name));
    }
  }
  source.append("};\n");
}

} // namespace vkcnn::codegen
