#include "./shared.hpp"
#include <fmt/format.h>

void vkcnn::codegen::define_shared(std::string_view var, Type type,
                                   std::optional<unsigned int> arraySize,
                                   std::string &source) {

  if (arraySize) {
    source.append(
        fmt::format("shared {} {}[{}];\n", type, var, arraySize.value()));
  } else {
    source.append(fmt::format("shared {} {};\n", type, var));
  }
}
