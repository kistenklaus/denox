#pragma once

#include "pyvk/host/Rational.hpp"
#include "pyvk/host/Tensor.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <glm/glm.hpp>
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
      std::vector<std::string> resultsOf;
    } concat;
    struct Output {
    } output;
    ~Infos() {} // manually managed.
    Infos() {}
  } info;

  ~LayerDescription() { reset(); }

  LayerDescription() : type(LayerType::None) {}

  LayerDescription(const LayerDescription &o)
      : name(o.name), type(o.type) {
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = o.info.input;
      break;
    case LayerType::Conv2d:
      info.conv2d = o.info.conv2d;
      break;
    case LayerType::Activation:
      info.activation = o.info.activation;
      break;
    case LayerType::MaxPool:
      info.maxPool = o.info.maxPool;
      break;
    case LayerType::Upsample:
      info.upsample = o.info.upsample;
      break;
    case LayerType::Concat:
      info.concat = o.info.concat;
      break;
    case LayerType::Output:
      info.output = o.info.output;
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
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = o.info.input;
      break;
    case LayerType::Conv2d:
      info.conv2d = o.info.conv2d;
      break;
    case LayerType::Activation:
      info.activation = o.info.activation;
      break;
    case LayerType::MaxPool:
      info.maxPool = o.info.maxPool;
      break;
    case LayerType::Upsample:
      info.upsample = o.info.upsample;
      break;
    case LayerType::Concat:
      info.concat = o.info.concat;
      break;
    case LayerType::Output:
      info.output = o.info.output;
      break;
    }
    return *this;
  }

  LayerDescription(LayerDescription &&o) : name(o.name), type(o.type) {
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = std::move(o.info.input);
      break;
    case LayerType::Conv2d:
      info.conv2d = std::move(o.info.conv2d);
      break;
    case LayerType::Activation:
      info.activation = std::move(o.info.activation);
      break;
    case LayerType::MaxPool:
      info.maxPool = std::move(o.info.maxPool);
      break;
    case LayerType::Upsample:
      info.upsample = std::move(o.info.upsample);
      break;
    case LayerType::Concat:
      info.concat = std::move(o.info.concat);
      break;
    case LayerType::Output:
      info.output = std::move(o.info.output);
      break;
    }
  }

  LayerDescription &operator=(LayerDescription &&o) {
    if (this == &o) {
      return *this;
    }
    reset();
    name = o.name;
    type = o.type;
    switch (type) {
    case LayerType::None:
      break;
    case LayerType::Input:
      info.input = std::move(o.info.input);
      break;
    case LayerType::Conv2d:
      info.conv2d =
          std::move(o.info.conv2d); // <- only one where it actually matters.
      break;
    case LayerType::Activation:
      info.activation = std::move(o.info.activation);
      break;
    case LayerType::MaxPool:
      info.maxPool = std::move(o.info.maxPool);
      break;
    case LayerType::Upsample:
      info.upsample = std::move(o.info.upsample);
      break;
    case LayerType::Concat:
      info.concat = std::move(o.info.concat);
      break;
    case LayerType::Output:
      info.output = std::move(o.info.output);
      break;
    }
    return *this;
  }

  // ======= NAMED constructors =============
  static LayerDescription input(const std::string &name,
                                   unsigned int channels) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Input;
    desc.info.input.channels = channels;
    return desc;
  }

  static LayerDescription conv2d(const std::string &name,
                                    const glm::uvec2 &kernelSize,
                                    const glm::uvec2 &stride,
                                    const glm::uvec2 &padding,
                                    Tensor<float, TensorFormat_OIHW> weights) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Conv2d;
    new (&desc.info.conv2d.kernelSize) glm::uvec2(kernelSize);
    new (&desc.info.conv2d.stride) glm::uvec2(stride);
    new (&desc.info.conv2d.padding) glm::uvec2(padding);
    new (&desc.info.conv2d.weights)
        Tensor<float, TensorFormat_OIHW>(std::move(weights));
    // TODO bias.
    return desc;
  }
  static LayerDescription activation(const std::string &name,
                                        const ActivationFunction func) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Activation;
    desc.info.activation.func = func;
    return desc;
  }

  static LayerDescription upsample(const std::string &name,
                                      const Rational &scaleFactor,
                                      const UpsampleFilterMode mode) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Upsample;
    desc.info.upsample.scaleFactor = scaleFactor;
    desc.info.upsample.filterMode = mode;
    return desc;
  }

  static LayerDescription maxPool(const std::string &name,
                                     const glm::uvec2 &kernelSize,
                                     const glm::uvec2 &stride) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::MaxPool;
    desc.info.maxPool.kernelSize = kernelSize;
    desc.info.maxPool.stride = stride;
    return desc;
  }
  static LayerDescription concat(const std::string &name,
                                    const std::span<std::string> &resultsOf) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Concat;
    desc.info.concat.resultsOf =
        std::vector(resultsOf.begin(), resultsOf.end());
    return desc;
  }

  static LayerDescription output(const std::string &name) {
    LayerDescription desc;
    desc.name = name;
    desc.type = LayerType::Output;
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

static constexpr auto x = sizeof(LayerDescription);

} // namespace pyvk
