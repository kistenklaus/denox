#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
namespace denox::compiler {

struct NamedExtent {
  memory::optional<memory::string> width;
  memory::optional<memory::string> height;
  memory::optional<memory::string> channels;
};

}
