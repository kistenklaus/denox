#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "vkcnn/dev/utils/merian.hpp"
#include <gtest/gtest.h>
#include <memory>
class TestEnv : public ::testing::Environment {
public:
  void SetUp() override {
    m_context = vkcnn::merian::createContext();
    m_shaderCompiler = std::make_shared<merian::SystemGlslcCompiler>(m_context);

    auto resources = m_context->get_extension<merian::ExtensionResources>();
    m_alloc = resources->resource_allocator();

    m_queue = m_context->get_queue_GCT();
  }

  void TearDown() override {
    m_context = nullptr;
    m_shaderCompiler = nullptr;
  }

  const merian::ContextHandle &ctx() const { return m_context; }
  const merian::ShaderCompilerHandle &sc() const { return m_shaderCompiler; }
  const merian::ResourceAllocatorHandle &alloc() const { return m_alloc; }
  const merian::QueueHandle &queue() const { return m_queue; }

private:
  merian::ContextHandle m_context;
  merian::ShaderCompilerHandle m_shaderCompiler;
  merian::ResourceAllocatorHandle m_alloc;
  merian::QueueHandle m_queue;
};

// Global accessor
extern TestEnv *env;
