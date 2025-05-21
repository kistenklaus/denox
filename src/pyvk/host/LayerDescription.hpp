#pragma once

#include "pyvk/host/Rational.hpp"
#include "pyvk/host/Tensor.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <glm/glm.hpp>
#include <print>
#include <string>
#include <utility>
namespace pyvk {

enum class LayerType {
  None,
  Input,
  Conv2d,
  Activation,
  MaxPool,
  Upsample,
  Concat,
  Output
};

enum class UpsampleFilterMode {
  Nearest,
};

enum class ActivationFunction { Relu };

struct LayerDescription {
  std::string name;
  LayerType type;
  std::string inputName;
  union Infos {
    struct Input {
      unsigned int channels;
    } input;
    struct Conv2d {
      glm::uvec2 kernelSize;
      glm::uvec2 stride;
      glm::uvec2 padding;
      Tensor<float, TensorFormat_OIHW> weights;
      // TODO bias pointer! (some datastructure which is not copyable)
    } conv2d;
    struct Activation {
      ActivationFunction func;
    } activation;
    struct MaxPool {
      glm::uvec2 kernelSize;
      glm::uvec2 stride;
      glm::uvec2 padding;
    } maxPool;
    struct Upsample {
      Rational scaleFactor;
      UpsampleFilterMode filterMode;
    } upsample;
    struct Concat {
      std::string to;
    } concat;
    struct Output {
    } output;
    ~Infos() {} // manually managed.
    Infos() {}
  } info;

  ~LayerDescription() { reset(); }

  LayerDescription() : type(LayerType::None) {}

  LayerDescription(const LayerDescription &o)
      : name(o.name), type(o.type), inputName(o.inputName) {
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      new (&info.input) Infos::Input(o.info.input);
      break;
    case LayerType::Conv2d:
      new (&info.conv2d) Infos::Conv2d(o.info.conv2d);
      break;
    case LayerType::Activation:
      new (&info.activation) Infos::Activation(o.info.activation);
      break;
    case LayerType::MaxPool:
      new (&info.maxPool) Infos::MaxPool(o.info.maxPool);
      break;
    case LayerType::Upsample:
      new (&info.upsample) Infos::Upsample(o.info.upsample);
      break;
    case LayerType::Concat:
      new (&info.concat) Infos::Concat(o.info.concat);
      break;
    case LayerType::Output:
      new (&info.output) Infos::Output(o.info.output);
      break;
    }
  }

  LayerDescription &operator=(const LayerDescription &o) {
    if (this == &o) {
      return *this;
    }
    reset();
    name = o.name;
    type = o.type;
    inputName = o.inputName;
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = Infos::Input(o.info.input);
      break;
    case LayerType::Conv2d:
      info.conv2d = Infos::Conv2d(o.info.conv2d);
      break;
    case LayerType::Activation:
      info.activation = Infos::Activation(o.info.activation);
      break;
    case LayerType::MaxPool:
      info.maxPool = Infos::MaxPool(o.info.maxPool);
      break;
    case LayerType::Upsample:
      info.upsample = Infos::Upsample(o.info.upsample);
      break;
    case LayerType::Concat:
      info.concat = Infos::Concat(o.info.concat);
      break;
    case LayerType::Output:
      info.output = Infos::Output(o.info.output);
      break;
    }
    return *this;
  }

  LayerDescription(LayerDescription &&o)
      : name(std::move(o.name)), type(o.type),
        inputName(std::move(o.inputName)) {
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      new (&info.input) Infos::Input(o.info.input);
      break;
    case LayerType::Conv2d:
      new (&info.conv2d) Infos::Conv2d(o.info.conv2d);
      break;
    case LayerType::Activation:
      new (&info.activation) Infos::Activation(o.info.activation);
      break;
    case LayerType::MaxPool:
      new (&info.maxPool) Infos::MaxPool(o.info.maxPool);
      break;
    case LayerType::Upsample:
      new (&info.upsample) Infos::Upsample(o.info.upsample);
      break;
    case LayerType::Concat:
      new (&info.concat) Infos::Concat(o.info.concat);
      break;
    case LayerType::Output:
      new (&info.output) Infos::Output(o.info.output);
      break;
    }
  }

  LayerDescription &operator=(LayerDescription &&o) {
    if (this == &o) {
      return *this;
    }
    reset();
    name = std::move(o.name);
    type = o.type;
    inputName = std::move(o.inputName);
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = Infos::Input(o.info.input);
      break;
    case LayerType::Conv2d:
      info.conv2d = Infos::Conv2d(o.info.conv2d);
      break;
    case LayerType::Activation:
      info.activation = Infos::Activation(o.info.activation);
      break;
    case LayerType::MaxPool:
      info.maxPool = Infos::MaxPool(o.info.maxPool);
      break;
    case LayerType::Upsample:
      info.upsample = Infos::Upsample(o.info.upsample);
      break;
    case LayerType::Concat:
      info.concat = Infos::Concat(o.info.concat);
      break;
    case LayerType::Output:
      info.output = Infos::Output(o.info.output);
      break;
    }    return *this;
  }

  // ======= NAMED constructors =============
  static LayerDescription input(const std::string &name,
                                unsigned int channels) {
    LayerDescription desc;
    desc.name = name;
    desc.inputName = "interface";
    desc.type = LayerType::Input;
    new (&desc.info.input.channels) int(channels);
    return desc;
  }

  static LayerDescription
  conv2d(const std::string &name, const std::string &input,
         const glm::uvec2 &kernelSize, const glm::uvec2 &stride,
         const glm::uvec2 &padding, Tensor<float, TensorFormat_OIHW> weights) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Conv2d;
    desc.inputName = input;
    new (&desc.info.conv2d.kernelSize) glm::uvec2(kernelSize);
    new (&desc.info.conv2d.stride) glm::uvec2(stride);
    new (&desc.info.conv2d.padding) glm::uvec2(padding);
    new (&desc.info.conv2d.weights)
        Tensor<float, TensorFormat_OIHW>(std::move(weights));
    // TODO bias.
    return desc;
  }
  static LayerDescription activation(const std::string &name,
                                     const std::string &input,
                                     const ActivationFunction func) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Activation;
    desc.inputName = input;
    new (&desc.info.activation.func) ActivationFunction(func);
    return desc;
  }

  static LayerDescription upsample(const std::string &name,
                                   const std::string &input,
                                   const Rational &scaleFactor,
                                   const UpsampleFilterMode mode) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Upsample;
    desc.inputName = input;
    new (&desc.info.upsample.scaleFactor) Rational(scaleFactor);
    new (&desc.info.upsample.filterMode) UpsampleFilterMode(mode);
    return desc;
  }

  static LayerDescription maxPool(const std::string &name,
                                  const std::string &input,
                                  const glm::uvec2 &kernelSize,
                                  const glm::uvec2 &stride) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::MaxPool;
    desc.inputName = input;
    new (&desc.info.maxPool.kernelSize) glm::uvec2(kernelSize);
    new (&desc.info.maxPool.stride) glm::uvec2(stride);
    return desc;
  }
  static LayerDescription concat(const std::string &name,
                                 const std::string &input,
                                 const std::string &to) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Concat;
    desc.inputName = input;
    new (&desc.info.concat.to) std::string(to);
    return desc;
  }

  static LayerDescription output(const std::string &name,
                                 const std::string &input) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Output;
    desc.inputName = input;
    return desc;
  }

private:
  void reset() {
    switch (type) {
    case LayerType::Input:
      info.input.~Input();
      break;
    case LayerType::Conv2d:
      info.conv2d.~Conv2d();
      break;
    case LayerType::Activation:
      info.activation.~Activation();
      break;
    case LayerType::MaxPool:
      info.maxPool.~MaxPool();
      break;
    case LayerType::Upsample:
      info.upsample.~Upsample();
      break;
    case LayerType::Concat:
      info.concat.~Concat();
      break;
    case LayerType::Output:
      info.output.~Output();
      break;
    case LayerType::None:
      // does nothing.
      break;
    }
    type = LayerType::None;
  }
};

} // namespace pyvk
