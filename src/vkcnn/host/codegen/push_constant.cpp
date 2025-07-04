#include "./push_constant.hpp"
#include <fmt/format.h>
#include <optional>

namespace vkcnn::codegen {

void push_constant(std::span<const Variable> entries, std::string_view name,
                   std::optional<std::string_view> var, std::string &source) {

  source.append(fmt::format("layout(push_constant) uniform {} {{\n", name));

  for (auto var : entries) {
    source.append(fmt::format("  {} {};\n", var.type, var.name));
  }

  source.push_back('}');
  if (var) {
    source.push_back(' ');
    source.append(var.value());
  }
  source.push_back(';');
  source.push_back('\n');
}
} // namespace vkcnn::codegen
