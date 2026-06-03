#include "version.hpp"
#include "denox/common/commit_hash.hpp"
#include "denox/common/version.hpp"
#include <iostream>

void version() {
  std::cout << "denox " << denox::version()
            << " (https://github.com/kistenklaus/denox.git at "
            << denox::commit_hash() << ")" << std::endl;
  std::cout << "Copyright (C) 2025-2026 Karl Sassie" << std::endl;
  std::cout << "This is free software; see the source for copying conditions."
            << std::endl;
  std::cout << "There is NO warranty; not even for MERCHANTABILITY or FITNESS "
               "FOR A PARTICULAR PURPOSE.\n"
            << std::endl;
}
