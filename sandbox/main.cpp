#include "denox/runtime.hpp"
#include <fmt/printf.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main() {

  denox::RuntimeContext context;
  if (denox::create_runtime_context("*RTX*", &context) < 0) {
    throw std::runtime_error("Failed to create context.");
  }

  std::ifstream dnxFile("./net.dnx", std::ios::binary);
  std::vector<std::uint8_t> dnxBinary((std::istreambuf_iterator<char>(dnxFile)),
                                      std::istreambuf_iterator<char>());

  denox::RuntimeModel model;
  if (denox::create_runtime_model(context,
                                  static_cast<const void *>(dnxBinary.data()),
                                  dnxBinary.size(), &model) < 0) {
    throw std::runtime_error("Failed to create runtime model");
  }

  denox::DynamicExtent extents[2];
  extents[0].name = "H";
  extents[0].value = 1080;
  extents[1].name = "W";
  extents[1].value = 1920;
  denox::RuntimeModelInstance instance;
  if (denox::create_runtime_model_instance(context, model, 2, extents,
                                           &instance)) {
    throw std::runtime_error("Failed to create runtime model instance.");
  }

  std::vector<std::uint16_t> input(1080 * 1920 * 3);
  void *pinput = input.data();

  denox::EvalResult result =
      denox::eval_runtime_model_instance(context, instance, 1, &pinput);

  denox::destroy_eval_result(result);

  denox::destroy_runtime_model_instance(context, instance);

  denox::destroy_runtime_model(context, model);

  denox::destroy_runtime_context(context);
}
