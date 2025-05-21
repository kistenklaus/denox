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

  void conv2d(const std::string &name, Tensor<float, TensorFormat_OIHW> weights,
              const glm::uvec2 padding, const glm::uvec2 &stride) {

    const auto &weightShape = weights.shape();

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::conv2d(name,
                                 glm::uvec2(weightShape[3], weightShape[2]),
                                 stride, padding, std::move(weights))));
  }

  void activation(const std::string name, ActivationFunction func) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::activation(name, func)));
  }

  void maxPool(const std::string &name, const glm::uvec2 kernelSize,
               const glm::uvec2 stride) {
    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::maxPool(name, kernelSize, stride)));
  }

  void upsample(const std::string &name, const Rational scaleFactor,
                UpsampleFilterMode mode) {

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::upsample(name, scaleFactor, mode)));
  }

  void concat(const std::string &name, std::span<std::string> &resultsOf) {

    m_layers.push_back(std::make_shared<LayerDescription>(
        LayerDescription::concat(name, resultsOf)));
  }

  void output(const std::string &name) {
    m_layers.push_back(
        std::make_shared<LayerDescription>(LayerDescription::output(name)));
  }

  void logPretty() {
    for (const auto &layer : m_layers) {
      switch (layer->type) {
      case LayerType::None:
        std::print("NOOP -> ");
        break;
      case LayerType::Input:
        std::print("Input(WxHx{}) -> ", layer->info.input.channels);
        break;
      case LayerType::Conv2d:
        std::print("Conv2d({}) -> ", layer->info.conv2d.weights.shape()[0]);
        break;
      case LayerType::Activation:
        switch (layer->info.activation.func) {
        case ActivationFunction::Relu:
          std::print("Relu -> ");
          break;
        }
        break;
      case LayerType::MaxPool:
        std::print("MaxPool -> ");
        break;
      case LayerType::Upsample:
        std::print("Upsample({}/{}) -> ", layer->info.upsample.scaleFactor.num,
                   layer->info.upsample.scaleFactor.den);
        break;
      case LayerType::Concat:
        std::print("Concat -> ");
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
