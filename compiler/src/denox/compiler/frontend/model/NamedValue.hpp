#pragma once

#include "denox/symbolic/Sym.hpp"
#include <string>

namespace denox {

struct NamedValue {
  std::string name;
  Sym value;
  bool imported;
};

}
