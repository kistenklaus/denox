#include "denox/runtime.hpp"
#include "f16.hpp"
#include <cassert>
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

  assert(denox::get_runtime_model_input_count(model) == 1);
  assert(denox::get_runtime_model_output_count(model) == 1);
  denox::DynamicExtent extents[2];
  std::size_t inW = 16;
  std::size_t inH = 8;
  extents[0].name = "W";
  extents[0].value = inW;
  extents[1].name = "H";
  extents[1].value = inH;
  denox::RuntimeModelInstance instance;
  if (denox::create_runtime_model_instance(context, model, 2, extents,
                                           &instance)) {
    throw std::runtime_error("Failed to create runtime model instance.");
  }
  std::size_t inCh;
  std::size_t checkInW;
  std::size_t checkInH;
  denox::get_runtime_model_instance_input_shape(instance, 0, &checkInW,
                                                &checkInH, &inCh);
  assert(checkInW == inW);
  assert(checkInH == inH);

  std::vector<f16> input(inW * inH * inCh * sizeof(f16));
  void *pinputs = input.data();

  std::size_t outW;
  std::size_t outH;
  std::size_t outCh;
  denox::get_runtime_model_instance_output_shape(instance, 0, &outW, &outH,
                                                 &outCh);

  std::vector<f16> output(outW * outH * outCh);
  std::size_t expectedOutSize;
  denox::get_runtime_model_instance_output_byte_size(instance, 0,
                                                     &expectedOutSize);

  assert(output.size() * sizeof(decltype(output)::value_type) ==
         expectedOutSize);
  void *poutputs = output.data();

  denox::eval_runtime_model_instance(context, instance, &pinputs, &poutputs);

  for (std::size_t c = 0; c < outCh; ++c) {
    fmt::println("Channel: {}", c);
    for (std::size_t h = 0; h < outH; ++h) {
      for (std::size_t w = 0; w < outW; ++w) {
        std::size_t index = h * (outW * outCh) + w * (outCh) + c;
        fmt::print(" {:^7.3f} ", static_cast<float>(output[index]));
      }
      fmt::println("");
    }
  }

  denox::destroy_runtime_model_instance(context, instance);

  denox::destroy_runtime_model(context, model);

  denox::destroy_runtime_context(context);
}
