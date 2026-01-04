#pragma once

#include "denox/memory/container/string.hpp"
#include "denox/symbolic/Sym.hpp"

namespace denox {

struct ValueName {
  memory::string name; 
  Sym value;
};

}

