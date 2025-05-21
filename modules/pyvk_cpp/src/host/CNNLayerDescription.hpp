#pragma once

#include "src/host/Tensor.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <glm/glm.hpp>
#include <string>
#include <utility>
namespace pyvk {

enum class CNNLayerType {
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

struct CNNLayerDescription {
  std::string name;
  CNNLayerType type;
  union Infos {
    struct Input {
    } input;
    struct Conv2d {
      TensorShape<2> kernelSize;
      TensorShape<2> stride;
      TensorShape<2> padding;
      Tensor<float, TensorFormat_OIHW> weights;
      // TODO bias pointer! (some datastructure which is not copyable)
    } conv2d;
    struct Activation {
      ActivationFunction func;
    } activation;
    struct MaxPool {
      TensorShape<2> kernelSize;
      TensorShape<2> stride;
    } maxPool;
    struct Upsample {
      UpsampleFilterMode filterMode;
    } upsample;
    struct Concat {
      // TODO!
    } concat;
    struct Output {
    } output;
    ~Infos() {} // manually managed.
    Infos() {}
  } info;

  ~CNNLayerDescription() { reset(); }

  CNNLayerDescription() : type(CNNLayerType::None) {}

  CNNLayerDescription(const CNNLayerDescription &o)
      : name(o.name), type(o.type), inputShape(o.inputShape),
        outputShape(o.outputShape) {
    switch (type) {
    case CNNLayerType::None:
      break;
    case CNNLayerType::Input:
      info.input = o.info.input;
      break;
    case CNNLayerType::Conv2d:
      info.conv2d = o.info.conv2d;
      break;
    case CNNLayerType::Activation:
      info.activation = o.info.activation;
      break;
    case CNNLayerType::MaxPool:
      info.maxPool = o.info.maxPool;
      break;
    case CNNLayerType::Upsample:
      info.upsample = o.info.upsample;
      break;
    case CNNLayerType::Concat:
      info.concat = o.info.concat;
      break;
    case CNNLayerType::Output:
      info.output = o.info.output;
      break;
    }
  }

  CNNLayerDescription &operator=(const CNNLayerDescription &o) {
    if (this == &o) {
      return *this;
    }
    reset();
    inputShape = o.inputShape;
    outputShape = o.outputShape;
    name = o.name;
    switch (type) {
    case CNNLayerType::None:
      break;
    case CNNLayerType::Input:
      info.input = o.info.input;
      break;
    case CNNLayerType::Conv2d:
      info.conv2d = o.info.conv2d;
      break;
    case CNNLayerType::Activation:
      info.activation = o.info.activation;
      break;
    case CNNLayerType::MaxPool:
      info.maxPool = o.info.maxPool;
      break;
    case CNNLayerType::Upsample:
      info.upsample = o.info.upsample;
      break;
    case CNNLayerType::Concat:
      info.concat = o.info.concat;
      break;
    case CNNLayerType::Output:
      info.output = o.info.output;
      break;
    }
    return *this;
  }

  CNNLayerDescription(CNNLayerDescription &&o)
      : name(o.name), type(o.type), inputShape(std::move(o.inputShape)),
        outputShape(std::move(o.outputShape)) {
    switch (type) {
    case CNNLayerType::None:
      break;
    case CNNLayerType::Input:
      info.input = std::move(o.info.input);
      break;
    case CNNLayerType::Conv2d:
      info.conv2d = std::move(o.info.conv2d);
      break;
    case CNNLayerType::Activation:
      info.activation = std::move(o.info.activation);
      break;
    case CNNLayerType::MaxPool:
      info.maxPool = std::move(o.info.maxPool);
      break;
    case CNNLayerType::Upsample:
      info.upsample = std::move(o.info.upsample);
      break;
    case CNNLayerType::Concat:
      info.concat = std::move(o.info.concat);
      break;
    case CNNLayerType::Output:
      info.output = std::move(o.info.output);
      break;
    }
  }

  CNNLayerDescription &operator=(CNNLayerDescription &&o) {
    if (this == &o) {
      return *this;
    }
    reset();
    type = o.type;
    inputShape = o.inputShape;
    outputShape = o.outputShape;
    name = o.name;
    switch (type) {
    case CNNLayerType::None:
      break;
    case CNNLayerType::Input:
      info.input = std::move(o.info.input);
      break;
    case CNNLayerType::Conv2d:
      info.conv2d =
          std::move(o.info.conv2d); // <- only one where it actually matters.
      break;
    case CNNLayerType::Activation:
      info.activation = std::move(o.info.activation);
      break;
    case CNNLayerType::MaxPool:
      info.maxPool = std::move(o.info.maxPool);
      break;
    case CNNLayerType::Upsample:
      info.upsample = std::move(o.info.upsample);
      break;
    case CNNLayerType::Concat:
      info.concat = std::move(o.info.concat);
      break;
    case CNNLayerType::Output:
      info.output = std::move(o.info.output);
      break;
    }
    return *this;
  }

  // ======= NAMED constructors =============
  static CNNLayerDescription input(const std::string &name,
                                   const TensorShape<3> &inputShape) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Input;
    desc.inputShape = inputShape;
    desc.outputShape = inputShape;
    desc.info.input = {};
    return desc;
  }

  static CNNLayerDescription
  conv2d(const std::string &name, const TensorShape<3> &inputShape,
         const TensorShape<3> &outputShape, const TensorShape<2> &kernelSize,
         const TensorShape<2> &stride, const TensorShape<2> &padding,
         Tensor<float, TensorFormat_OIHW> &&weights) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Conv2d;
    desc.inputShape = inputShape;
    desc.outputShape = outputShape;
    desc.info.conv2d.kernelSize = kernelSize;
    desc.info.conv2d.stride = stride;
    desc.info.conv2d.padding = padding;
    desc.info.conv2d.weights = std::move(weights);
    // TODO bias.
    return desc;
  }
  static CNNLayerDescription activation(const std::string &name,
                                        const TensorShape<3> &inputShape,
                                        const ActivationFunction func) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Activation;
    desc.inputShape = inputShape;
    desc.outputShape = inputShape;
    desc.info.activation.func = func;
    return desc;
  }

  static CNNLayerDescription upsample(const std::string &name,
                                      const TensorShape<3> &inputShape,
                                      const TensorShape<3> &outputShape,
                                      const UpsampleFilterMode mode) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Upsample;
    desc.inputShape = inputShape;
    desc.outputShape = inputShape;
    desc.info.upsample.filterMode = mode;
    return desc;
  }

  static CNNLayerDescription maxPool(const std::string &name,
                                     const TensorShape<3> &inputShape,
                                     const TensorShape<3> &outputShape,
                                     const TensorShape<2> &kernelSize,
                                     const TensorShape<2> &stride) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::MaxPool;
    desc.inputShape = inputShape;
    desc.outputShape = outputShape;
    desc.info.maxPool.kernelSize = kernelSize;
    desc.info.maxPool.stride = stride;
    return desc;
  }
  static CNNLayerDescription concat(const std::string &name,
                                    const TensorShape<3> &inputShape,
                                    const TensorShape<3> &outputShape) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Concat;
    desc.inputShape = inputShape;
    desc.outputShape = outputShape;
    desc.info.concat = {};
    return desc;
  }

  static CNNLayerDescription output(const std::string &name,
                                    const TensorShape<3> &inputShape) {
    CNNLayerDescription desc;
    desc.name = name;
    desc.type = CNNLayerType::Output;
    desc.inputShape = inputShape;
    desc.outputShape = inputShape;
    desc.info.concat = {};
    return desc;
  }

private:
  void reset() {
    switch (type) {
    case CNNLayerType::Input:
      info.input.~Input();
      break;
    case CNNLayerType::Conv2d:
      info.conv2d.~Conv2d();
      break;
    case CNNLayerType::Activation:
      info.activation.~Activation();
      break;
    case CNNLayerType::MaxPool:
      info.maxPool.~MaxPool();
      break;
    case CNNLayerType::Upsample:
      info.upsample.~Upsample();
      break;
    case CNNLayerType::Concat:
      info.concat.~Concat();
      break;
    case CNNLayerType::Output:
      info.output.~Output();
      break;
    case CNNLayerType::None:
      // does nothing.
      break;
    }
    type = CNNLayerType::None;
  }
};

static constexpr auto x = sizeof(CNNLayerDescription);

} // namespace pyvk
