#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fmt/printf.h>
#include <span>
#include <stdexcept>
#include <vma.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

struct Buffer {
  VkBuffer buffer;
  VmaAllocation allocation;
};

class Context {
public:
  explicit Context(const char *deviceName);
  ~Context();
  Context(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(const Context &) = delete;
  Context &operator=(Context &&) = delete;

  Buffer createBuffer(std::size_t size, VkBufferUsageFlags usage,
                      VmaAllocationCreateFlags flags) {
    VkBufferCreateInfo bufferInfo;
    std::memset(&bufferInfo, 0, sizeof(VkBufferCreateInfo));
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.flags = flags;
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    Buffer buffer;
    VkResult result =
        vmaCreateBuffer(m_vma, &bufferInfo, &allocInfo, &buffer.buffer,
                        &buffer.allocation, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create buffer.");
    }
    return buffer;
  }

  void destroyBuffer(Buffer buffer) {
    assert(buffer.buffer != VK_NULL_HANDLE);
    assert(buffer.allocation != VK_NULL_HANDLE);
    vmaDestroyBuffer(m_vma, buffer.buffer, buffer.allocation);
  }

  VkDescriptorSetLayout createDescriptorSetLayout(
      std::span<const VkDescriptorSetLayoutBinding> bindings) {
    VkDescriptorSetLayoutCreateInfo layoutInfo;
    std::memset(&layoutInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pBindings = bindings.data();
    layoutInfo.bindingCount = bindings.size();

    VkDescriptorSetLayout layout;
    {
      VkResult result =
          vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
      }
    }
    return layout;
  }

  void destroyDescriptorSetLayout(VkDescriptorSetLayout layout) {
    assert(layout != VK_NULL_HANDLE);
    vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
  }

  VkPipelineLayout
  createPipelineLayout(std::span<const VkDescriptorSetLayout> descriptorLayouts,
                       std::uint32_t pushConstantRange) {
    VkPipelineLayoutCreateInfo layoutInfo;
    std::memset(&layoutInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = descriptorLayouts.size();
    layoutInfo.pSetLayouts = descriptorLayouts.data();
    VkPushConstantRange pushConstant;
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.size = pushConstantRange;
    pushConstant.offset = 0;
    layoutInfo.pPushConstantRanges = &pushConstant;
    layoutInfo.pushConstantRangeCount = 1;

    VkPipelineLayout layout;
    {
      VkResult result =
          vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &layout);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout.");
      }
    }
    return layout;
  }

  void destroyPipelineLayout(VkPipelineLayout layout) {
    assert(layout != VK_NULL_HANDLE);
    vkDestroyPipelineLayout(m_device, layout, nullptr);
  }

  VkPipeline createComputePipeline(VkPipelineLayout layout,
                                   std::span<const std::uint32_t> binary,
                                   const char *entry) {
    assert(layout != nullptr);
    assert(!binary.empty());
    assert(entry != nullptr);
    VkShaderModule module;
    {
      VkShaderModuleCreateInfo shaderInfo;
      std::memset(&shaderInfo, 0, sizeof(VkShaderModuleCreateInfo));
      shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shaderInfo.pCode = binary.data();
      shaderInfo.codeSize = binary.size() * sizeof(std::uint32_t);
      VkResult result =
          vkCreateShaderModule(m_device, &shaderInfo, nullptr, &module);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module.");
      }
    }

    VkComputePipelineCreateInfo pipelineInfo;
    std::memset(&pipelineInfo, 0, sizeof(VkComputePipelineCreateInfo));
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = entry;

    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    {
      VkResult result = vkCreateComputePipelines(
          m_device, nullptr, 1, &pipelineInfo, nullptr, &pipeline);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline.");
      }
    }
    vkDestroyShaderModule(m_device, module, nullptr);
    return pipeline;
  }

  void destroyPipeline(VkPipeline pipeline) {
    assert(pipeline != VK_NULL_HANDLE);
    vkDestroyPipeline(m_device, pipeline, nullptr);
  }

private:
  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  std::uint32_t m_queueFamily;
  VkQueue m_queue;
  VmaAllocator m_vma;
};

} // namespace denox::runtime
