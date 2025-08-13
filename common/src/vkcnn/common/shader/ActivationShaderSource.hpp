#pragma once

#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/shader/ShaderDefines.hpp"
#include "vkcnn/common/shader/ShaderLang.hpp"
#include "vkcnn/common/shader/SpecializationConstants.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cstddef>
#include <glm/vec3.hpp>
#include <memory>
#include <string>
#include <vector>
namespace vkcnn {

class ActivationShaderSource {
public:
  struct DebugInfo {
    ActivationLayout inputLayout;
    FloatType inputType;

    ActivationLayout outputLayout;
    FloatType outputType;

    unsigned int channels;

    ActivationFunction function;

    std::string name;
  };

private:
  struct Storage {

    std::vector<std::byte> src;
    ShaderLang lang;

    SpecializationConstants specConstants;
    ShaderDefines defines;
    glm::uvec3 tileSize;
#ifndef NDEBUG
    DebugInfo debugInfo;
#endif
  };

public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationShaderSource(std::vector<std::byte> src, ShaderLang lang,
                                  SpecializationConstants specConstants,
                                  ShaderDefines defines, glm::uvec3 tileSize,
                                  const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(
            alloc, std::move(src), lang, std::move(specConstants),
            std::move(defines), tileSize,
            DebugInfo{ActivationLayout::HWC, FloatType::F16,
                      ActivationLayout::HWC, FloatType::F16, 0,
                      ActivationFunction::ReLU, "activation-no-debug-info"})) {}

  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationShaderSource(std::vector<std::byte> src, ShaderLang lang,
                                  SpecializationConstants specConstants,
                                  ShaderDefines defines, glm::uvec3 tileSize,
                                  ActivationLayout inputLayout,
                                  FloatType inputType,
                                  ActivationLayout outputLayout,
                                  FloatType outputType, unsigned int channels,
                                  ActivationFunction function, std::string name,
                                  const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(
            alloc, std::move(src), lang, std::move(specConstants),
            std::move(defines), tileSize,
            DebugInfo{inputLayout, inputType, outputLayout, outputType,
                      channels, function, std::move(name)})) {}

  std::span<const std::byte> src() const { return m_store->src; }
  ShaderLang lang() const { return m_store->lang; }
  const SpecializationConstants &specConstants() const {
    return m_store->specConstants;
  }
  const ShaderDefines &defines() const { return m_store->defines; }
  glm::uvec3 tileSize() const { return m_store->tileSize; }

#ifndef NDEBUG
  const DebugInfo &debugInfo() const { return m_store->debugInfo; }
#endif

private:
  std::shared_ptr<Storage> m_store;
};

} // namespace vkcnn
