#include "pyvk/host/Tensor.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <print>
#include <pyvk/host/NetworkDescription.hpp>

int main() {

  pyvk::NetworkDescription desc("input", 3);

  pyvk::Tensor<float, pyvk::TensorFormat_OIHW> conv0Weights({3, 3, 3, 3});
  desc.conv2d("conv0", std::move(conv0Weights), glm::uvec2(1), glm::uvec2(1));
  desc.activation("relu0", pyvk::ActivationFunction::Relu);

  pyvk::Tensor<float, pyvk::TensorFormat_OIHW> conv1Weights({3, 3, 3, 3});
  desc.conv2d("conv1", std::move(conv1Weights), glm::uvec2(1), glm::uvec2(1));
  desc.activation("relu1", pyvk::ActivationFunction::Relu);

  desc.output("output");


  desc.logPretty();

}
