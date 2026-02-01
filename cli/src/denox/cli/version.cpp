#include "version.hpp"
#include "denox/common/version.hpp"
#include <iostream>

void version() { std::cerr << "denox " << denox::version() << std::endl; }
