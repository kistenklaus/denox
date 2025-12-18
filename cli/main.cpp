

#include "parser/parse.hpp"
#include <fmt/printf.h>
int main(int argc, char **argv) {

  try {
    parse_argv(argc, argv);

  } catch (const std::exception& e) {
    fmt::println("{}", e.what());
  }
}
