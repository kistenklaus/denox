#pragma once

#include "memory/container/optional.hpp"
#include "memory/container/string.hpp"
namespace denox::compiler {

struct ModelMeta {
  memory::optional<memory::string> producerName;
  memory::optional<memory::string> producerVersion;
  memory::optional<memory::string> domain;
  memory::optional<memory::string> modelVersion;
};

} // namespace denox::compiler
