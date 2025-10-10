#include "denox/runtime.hpp"
#include <fmt/printf.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main() {

  denox::RuntimeContext context;
  if (denox::create_runtime_context(nullptr, &context) < 0) {
    throw std::runtime_error("Failed to create context.");
  }

  std::ifstream dnxFile("./net.dnx", std::ios::binary);
  std::vector<std::uint8_t> dnxBinary((std::istreambuf_iterator<char>(dnxFile)),
                                      std::istreambuf_iterator<char>());

  denox::RuntimeModel model;
  if (denox::create_runtime_model(context, static_cast<const void *>(dnxBinary.data()),
                                  dnxBinary.size(), &model) < 0) {
    throw std::runtime_error("Failed to create runtime model");
  }

  denox::destroy_runtime_model(context, model);

  denox::destroy_runtime_context(context);
}
