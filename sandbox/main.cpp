#include "denox/common/types.hpp"
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
  denox::Extent extents[2];
  std::size_t inW = 5;
  std::size_t inH = 5;
  extents[0].name = "W";
  extents[0].value = inW;
  extents[1].name = "H";
  extents[1].value = inH;
  denox::RuntimeInstance instance;

  if (denox::create_runtime_instance(context, model, 2, extents, &instance)) {
    throw std::runtime_error("Failed to create runtime model instance.");
  }
  denox::Extent checkInCh;
  denox::Extent checkInW;
  denox::Extent checkInH;
  denox::get_runtime_instance_tensor_shape(instance, "input", &checkInH,
                                           &checkInW, &checkInCh);
  assert(checkInW.value == inW);
  assert(checkInH.value == inH);
  std::size_t inCh = checkInCh.value;

  std::vector<f16> input(inW * inH * inCh * sizeof(f16));
  std::size_t it = 0;
  for (auto &x : input) {
    x = f16(static_cast<float>(++it));
  }

  fmt::println("INPUT");
  for (std::size_t c = 0; c < inCh; ++c) {
    fmt::println("Channel: {}", c);
    for (std::size_t h = 0; h < inH; ++h) {
      for (std::size_t w = 0; w < inW; ++w) {
        std::size_t index = h * (inW * inCh) + w * (inCh) + c;
        fmt::print(" {:^7.3f} ", static_cast<float>(input[index]));
      }
      fmt::println("");
    }
  }

  const void *pinputs = input.data();
  //
  denox::Extent outExtentCh;
  denox::Extent outExtentW;
  denox::Extent outExtentH;
  denox::get_runtime_instance_tensor_shape(instance, "output", &outExtentH,
                                           &outExtentW, &outExtentCh);
  std::size_t outCh = outExtentCh.value;
  std::size_t outW = outExtentW.value;
  std::size_t outH = outExtentH.value;

  std::vector<f16> output(outW * outH * outCh);

  std::size_t expectedOutSize =
      denox::get_runtime_instance_tensor_byte_size(instance, "output");
  assert(output.size() * sizeof(decltype(output)::value_type) ==
         expectedOutSize);
  void *poutputs = output.data();

  denox::eval_runtime_instance(context, instance, &pinputs, &poutputs);

  fmt::println("OUTPUT");
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

  denox::destroy_runtime_instance(context, instance);

  denox::destroy_runtime_model(context, model);

  denox::destroy_runtime_context(context);
}
