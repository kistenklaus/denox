#include "version.hpp"
#include <iostream>

void version() {
  std::cerr << "denox " << DENOX_VERSION_MAJOR << "." << DENOX_VERSION_MINOR << "." << DENOX_VERSION_PATCH << std::endl;
}

