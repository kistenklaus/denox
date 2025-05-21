#include "pyvk/host/DispatchOp.hpp"
#include "pyvk/host/LayerDescription.hpp"
#include "pyvk/host/Rational.hpp"
#include "pyvk/host/Tensor.hpp"
#include "pyvk/host/layer_fusion.hpp"
#include "pyvk/host/memory_planning.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <print>
#include <pyvk/host/NetworkDescription.hpp>

int main() {

  pyvk::NetworkDescription network("input", 3);

  pyvk::Tensor<float, pyvk::TensorFormat_OIHW> enc0Weights({3, 3, 3, 3});
  network.conv2d("enc0", "input", std::move(enc0Weights), glm::uvec2(1),
                 glm::uvec2(1));
  network.activation("relu0", "enc0", pyvk::ActivationFunction::Relu);

  network.maxPool("pool0", "relu0", glm::uvec2(2, 2), glm::uvec2(2, 2));

  pyvk::Tensor<float, pyvk::TensorFormat_OIHW> conv0Weights({3, 3, 3, 3});
  network.conv2d("conv0", "pool0", conv0Weights, glm::uvec2(1), glm::uvec2(1));
  network.activation("relu1", "conv0", pyvk::ActivationFunction::Relu);


  network.upsample("upsample0", "conv0", pyvk::Rational(2, 1),
                   pyvk::UpsampleFilterMode::Nearest);

  network.concat("concat0", "upsample0", "pool0");

  // TODO build better example with all operaitons.
  pyvk::Tensor<float, pyvk::TensorFormat_OIHW> conv1Weights({3, 3, 3, 3});

  network.conv2d("dec1", "concat0", std::move(conv1Weights), glm::uvec2(1),
                 glm::uvec2(1));
  network.activation("relu2", "dec1", pyvk::ActivationFunction::Relu);
  network.output("output", "relu2");
  network.logPretty();

  // ============== Simple Layer fusion ==============
  // Works with incredibly simple rules, we just go can we?, then yes.
  std::vector<pyvk::DispatchOp> ops = pyvk::fuseLayers(network);
  std::println("{:=^100}", "Operations performed per dispatch:");
  for (const auto &op : ops) {
    op.logPretty();
  }

  // ============== Memory planning =================
  // Here we ofcause can't already do the full memory planning, but we can
  // determine required buffers and their lifetime.

  std::println("{:=^100}", "LiveBuffers per dispatch");
  pyvk::planMemory(ops);
}
