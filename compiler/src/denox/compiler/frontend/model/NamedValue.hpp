#pragma once

#include "denox/symbolic/Sym.hpp"
#include <string>
namespace denox::compiler {

struct NamedValue {
  std::string name;
  Sym value;
  bool imported;
};

}
