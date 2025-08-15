#pragma once

#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/shader/ShaderDefines.hpp"
#include "vkcnn/common/shader/ShaderLang.hpp"
#include "vkcnn/common/shader/SpecializationConstants.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cstddef>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>
namespace vkcnn {

class PoolShaderSource {
public:
#ifndef NDEBUG
  struct DebugInfo {
    ActivationLayout inputLayout;
    FloatType inputType;

    ActivationLayout outputLayout;
    FloatType outputType;

    unsigned int channels;

    glm::uvec2 kernelSize;
    glm::uvec2 stride;
    glm::uvec2 padding;

    PoolFunction poolingFunc;

    std::string name;
  };
#endif
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
  explicit PoolShaderSource(std::vector<std::byte> src, ShaderLang lang,
                               SpecializationConstants specConstants,
                               ShaderDefines defines, glm::uvec3 tileSize,
                               const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(alloc, std::move(src), lang,
                                              std::move(specConstants),
                                              std::move(defines), tileSize,
                                              DebugInfo{
                                                  ActivationLayout::HWC,
                                                  FloatType::F16,
                                                  ActivationLayout::HWC,
                                                  FloatType::F16,
                                                  0,
                                                  glm::uvec2(0, 0),
                                                  glm::uvec2(0, 0),
                                                  glm::uvec2(0, 0),
                                                  PoolFunction::Max,
                                                  "pool-no-debug-info"
                                              })) {}

#ifndef NDEBUG
  template <typename Alloc = std::allocator<std::byte>>
  explicit PoolShaderSource(std::vector<std::byte> src, ShaderLang lang,
                               SpecializationConstants specConstants,
                               ShaderDefines defines, glm::uvec3 tileSize,
                               DebugInfo debugInfo, const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(
            alloc, std::move(src), lang, std::move(specConstants),
            std::move(defines), tileSize, std::move(debugInfo))) {}
#endif

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
