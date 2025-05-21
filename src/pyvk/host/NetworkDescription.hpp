#pragma once

#include "pyvk/host/LayerDescription.hpp"
#include "pyvk/host/Tensor.hpp"
#include <memory>
#include <print>
#include <vector>
namespace pyvk {

struct NetworkDescription {

  explicit NetworkDescription(std::string inputLayerName,
                              unsigned int channels) {

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::input(inputLayerName, channels)));
  }

  void conv2d(const std::string &name, const std::string &input,
              Tensor<float, TensorFormat_OIHW> weights, glm::uvec2 padding,
              glm::uvec2 stride) {

    const auto &weightShape = weights.shape();

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::conv2d(name, input,
                                 glm::uvec2(weightShape[3], weightShape[2]),
                                 stride, padding, std::move(weights))));
  }

  void activation(const std::string name, const std::string &input,
                  ActivationFunction func) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::activation(name, input, func)));
  }

  void maxPool(const std::string &name, const std::string &input,
               glm::uvec2 kernelSize, glm::uvec2 stride) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::maxPool(name, input, kernelSize, stride)));
  }

  void upsample(const std::string &name, const std::string &input,
                const Rational scaleFactor, UpsampleFilterMode mode) {

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::upsample(name, input, scaleFactor, mode)));
  }

  void concat(const std::string &name, const std::string &input,
              const std::string &to) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::concat(name, input, to)));
  }

  void output(const std::string &name, const std::string &input) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::output(name, input)));
  }

  std::span<const std::shared_ptr<LayerDescription>> layers() const {
    return m_layers;
  }

  void logPretty() {
    std::println("{:=^100}", "Netork");
    for (const auto &layer : m_layers) {
      switch (layer->type) {
      case LayerType::None:
        std::println("NOOP -> ");
        break;
      case LayerType::Input:
        std::println("Input(WxHx{}) -> ", layer->info.input.channels);
        break;
      case LayerType::Conv2d:
        std::println("Conv2d({}) -> ", layer->info.conv2d.weights.shape()[0]);
        break;
      case LayerType::Activation:
        switch (layer->info.activation.func) {
        case ActivationFunction::Relu:
          std::println("Relu -> ");
          break;
        }
        break;
      case LayerType::MaxPool:
        std::println("MaxPool -> ");
        break;
      case LayerType::Upsample:
        std::println("Upsample({}/{}) -> ",
                     layer->info.upsample.scaleFactor.num,
                     layer->info.upsample.scaleFactor.den);
        break;
      case LayerType::Concat:
        std::println("Concat -> ");
        break;
      case LayerType::Output:
        std::println("Output");
        break;
      }
    }
  }

private:
  std::vector<std::shared_ptr<LayerDescription>> m_layers;
};

} // namespace pyvk
