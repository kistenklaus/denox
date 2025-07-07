#pragma once

#include "vkcnn/common/shader/ShaderDefines.hpp"
#include "vkcnn/common/shader/ShaderLang.hpp"
#include "vkcnn/common/shader/SpecializationConstants.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FilterLayout.hpp"
#include "vkcnn/common/tensor/FilterShape.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <vector>

namespace vkcnn {

namespace details {}

class ConvShaderSource {
private:
  struct Storage {
    std::vector<std::byte> src;
    ShaderLang lang;

    SpecializationConstants specConstants;
    ShaderDefines defines;

    glm::uvec2 tileSize;

    ActivationLayout inputLayout;
    FloatType inputType;
    ActivationLayout outputLayout;
    FloatType outputType;

    FilterDescriptor filterDescriptor;

    std::string name;
  };

public:
  template <typename Alloc = std::allocator<std::byte>>
  ConvShaderSource(std::vector<std::byte> src, ShaderLang lang,
                   SpecializationConstants specConstants, ShaderDefines defines,
                   glm::uvec2 tileSize, ActivationLayout inputLayout,
                   FloatType inputType, ActivationLayout outputLayout,
                   FloatType outputType, FilterDescriptor filterDescriptor,
                   std::string name, const Alloc &alloc = {}) {
    m_store = std::allocate_shared<Storage>(
        alloc, std::move(src), lang, std::move(specConstants),
        std::move(defines), tileSize, inputLayout, inputType, outputLayout,
        outputType, filterDescriptor, name);
  }

  std::span<const std::byte> src() const { return m_store->src; }
  ShaderLang lang() const { return m_store->lang; }
  const SpecializationConstants &specializationConstants() const {
    return m_store->specConstants;
  }
  const ShaderDefines &defines() const { return m_store->defines; }

  const glm::uvec2 &tileSize() const { return m_store->tileSize; }

  ActivationLayout inputLayout() const { return m_store->inputLayout; }
  FloatType inputType() const { return m_store->inputType; }
  ActivationLayout outputLayout() const { return m_store->outputLayout; }
  FloatType outputType() const { return m_store->outputType; }

  FilterDescriptor filterDesc() const { return m_store->filterDescriptor; }
  FilterLayout filterLayout() const { return m_store->filterDescriptor.layout; }
  const FilterShape &filterShape() const {
    return m_store->filterDescriptor.shape;
  }
  FloatType filterType() const { return m_store->filterDescriptor.type; }

  const std::string& name() const {
    return m_store->name;
  }

private:
  std::shared_ptr<Storage> m_store;
};

} // namespace vkcnn
