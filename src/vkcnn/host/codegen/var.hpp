#pragma once

#include "vkcnn/host/codegen/type.hpp"
#include <optional>
#include <span>
#include <string>
namespace vkcnn::codegen {

struct Variable {
  Type type;
  std::string name;
  std::optional<std::size_t> arraySize = std::nullopt;
};

}; // namespace vkcnn::codegen
