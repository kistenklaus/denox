#pragma once

#include "vkcnn/host/DynamicWeightTensor.hpp"
#include "vkcnn/host/WeightTensorLayout.hpp"
#include "vkcnn/host/fprec.hpp"
#include "vkcnn/host/shaderlang.hpp"
#include <glm/fwd.hpp>
#include <glm/vec2.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
namespace vkcnn::host::conv2d {

class Conv2dSource {
private:
  struct Kernel {
    explicit Kernel(const DynamicWeightTensor &tensor)
        : m_variant(OutlinedKernel{tensor}) {}

    explicit Kernel(DynamicWeightTensor &&tensor)
        : m_variant(OutlinedKernel{std::move(tensor)}) {}

    explicit Kernel(glm::uvec2 kernelSize, unsigned int inputChannels,
                    unsigned int outputChannels, FPrec precision)
        : m_variant(InlinedKernel{kernelSize, inputChannels, outputChannels,
                                  precision}) {}

    glm::uvec2 kernelSize() const {
      const auto visitor = overloads{
          [](const OutlinedKernel &k) {
            return glm::uvec2(k.weightTensor.s(), k.weightTensor.r());
          },
          [](const InlinedKernel &k) { return k.kernelSize; },
      };
      return std::visit(visitor, m_variant);
    }

    FPrec prec() const {
      const auto visitor = overloads{
          [](const OutlinedKernel &k) { return k.weightTensor.precision(); },
          [](const InlinedKernel &k) { return k.precision; },
      };
      return std::visit(visitor, m_variant);
    }

    unsigned int inputChannels() const {
      const auto visitor = overloads{
          [](const OutlinedKernel &k) {
            return static_cast<unsigned int>(k.weightTensor.c());
          },
          [](const InlinedKernel &k) { return k.inputChannels; },
      };
      return std::visit(visitor, m_variant);
    }

    unsigned int outputChannels() const {
      const auto visitor = overloads{
          [](const OutlinedKernel &k) {
            return static_cast<unsigned int>(k.weightTensor.k());
          },
          [](const InlinedKernel &k) { return k.outputChannels; },
      };
      return std::visit(visitor, m_variant);
    }

    bool inlinedWeights() const {
      return std::holds_alternative<InlinedKernel>(m_variant);
    }

    std::optional<std::span<const std::byte>> weights() const {
      const auto visitor = overloads{
          [](const OutlinedKernel &k) {
            return std::optional<std::span<const std::byte>>(
                k.weightTensor.bufferView());
          },
          [](const InlinedKernel &) {
            return std::optional<std::span<const std::byte>>{};
          },
      };
      return std::visit(visitor, m_variant);
    }

    // May throw if it's not a outlined kernel (i.e. inlinedWeights == false)
    DynamicWeightTensor &weightTensor() {
      return std::get<OutlinedKernel>(m_variant).weightTensor;
    }

    const DynamicWeightTensor &weightTensor() const {
      return std::get<OutlinedKernel>(m_variant).weightTensor;
    }

  private:
    template <class... Ts> struct overloads : Ts... {
      using Ts::operator()...;
    };

    struct OutlinedKernel {
      DynamicWeightTensor weightTensor;
    };
    struct InlinedKernel {
      glm::uvec2 kernelSize;
      unsigned int inputChannels;
      unsigned int outputChannels;
      FPrec precision;
    };
    std::variant<OutlinedKernel, InlinedKernel> m_variant;
  };

public:
  Conv2dSource(ShaderLang lang, std::vector<std::byte> src, glm::uvec2 tileSize,
               const DynamicWeightTensor &kernel, glm::uvec2 stride,
               glm::uvec2 padding, FPrec inputPrec = FPrec::F32,
               FPrec outputPrec = FPrec::F32,
               std::string_view debugName = "conv3d-src")
      : m_lang(lang), m_source(std::move(src)), m_tileSize(tileSize),
        m_kernel(kernel), m_stride(stride), m_padding(padding),
        m_inputPrecision(inputPrec), m_outputPrecision(outputPrec),
        m_debugName(debugName) {}

  Conv2dSource(ShaderLang lang, std::vector<std::byte> src, glm::uvec2 tileSize,
               DynamicWeightTensor &&kernel, glm::uvec2 stride,
               glm::uvec2 padding, FPrec inputPrec = FPrec::F32,
               FPrec outputPrec = FPrec::F32,
               std::string_view debugName = "conv3d-src")
      : m_lang(lang), m_source(std::move(src)), m_tileSize(tileSize),
        m_kernel(kernel), m_stride(stride), m_padding(padding),
        m_inputPrecision(inputPrec), m_outputPrecision(outputPrec),
        m_debugName(debugName) {}

  Conv2dSource(ShaderLang lang, std::vector<std::byte> src, glm::uvec2 tileSize,
               glm::uvec2 kernelSize, unsigned int inputChannels,
               unsigned int outputChannels, FPrec weightPrec, glm::uvec2 stride,
               glm::uvec2 padding, FPrec inputPrec = FPrec::F32,
               FPrec outputPrec = FPrec::F32,
               std::string_view debugName = "conv3d-src")
      : m_lang(lang), m_source(std::move(src)), m_tileSize(tileSize),
        m_kernel(kernelSize, inputChannels, outputChannels, weightPrec),
        m_stride(stride), m_padding(padding), m_inputPrecision(inputPrec),
        m_outputPrecision(outputPrec), m_debugName(debugName) {}

  /// Language of the source code.
  ShaderLang lang() const { return m_lang; }

  /// Source code of the shader
  std::span<const std::byte> src() const { return m_source; }

  /// Size of a tile (i.e. workgroup).
  glm::uvec2 tileSize() const { return m_tileSize; }

  /// Precision of the input tensor.
  FPrec inputPrec() const { return m_inputPrecision; }

  /// Precision of the output tensor.
  FPrec outputPrec() const { return m_outputPrecision; }

  // /// Precision of the weight tensor.
  // /// As no effect if weights() returns std::nullopt.
  FPrec weightPrec() const { return m_kernel.prec(); };

  std::optional<std::span<const std::byte>> weights() const {
    return m_kernel.weights();
  }
  /// May throw if inlinedWeights() returns true.
  DynamicWeightTensor& weightTensor() { return m_kernel.weightTensor(); }
  /// May throw if inlinedWeights() returns true.
  const DynamicWeightTensor& weightTensor() const { return m_kernel.weightTensor(); }
  //
  // /// Amount of input channels.
  unsigned int inputChannels() const { return m_kernel.inputChannels(); }
  //
  // /// Amount of output channels.
  unsigned int outputChannels() const { return m_kernel.outputChannels(); }
  //
  glm::uvec2 kernelSize() const { return m_kernel.kernelSize(); }

  glm::uvec2 stride() const { return m_stride; }

  glm::uvec2 padding() const { return m_padding; }

  bool inlinedWeights() const { return m_kernel.inlinedWeights(); }

  std::string_view debugName() const { return m_debugName; }

private:
  ShaderLang m_lang;
  std::vector<std::byte> m_source;
  glm::uvec2 m_tileSize;

  Kernel m_kernel;
  glm::uvec2 m_stride;
  glm::uvec2 m_padding;

  FPrec m_inputPrecision;
  FPrec m_outputPrecision;

  std::string m_debugName;
};

struct Temp {};

} // namespace vkcnn::host::conv2d
