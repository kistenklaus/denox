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
                      VmaAllocationCreateFlags flags);
  void destroyBuffer(Buffer buffer);

  VkDescriptorSetLayout createDescriptorSetLayout(
      std::span<const VkDescriptorSetLayoutBinding> bindings);
  void destroyDescriptorSetLayout(VkDescriptorSetLayout layout);

  VkPipelineLayout
  createPipelineLayout(std::span<const VkDescriptorSetLayout> descriptorLayouts,
                       std::uint32_t pushConstantRange);
  void destroyPipelineLayout(VkPipelineLayout layout);

  VkPipeline createComputePipeline(VkPipelineLayout layout,
                                   std::span<const std::uint32_t> binary,
                                   const char *entry);
  void destroyPipeline(VkPipeline pipeline);

  VkCommandPool createCommandPool();
  void destroyCommandPool(VkCommandPool cmdPool);

  VkCommandBuffer allocCommandBuffer(VkCommandPool cmdPool);
  void freeCommandBuffer(VkCommandPool cmdPool, VkCommandBuffer cmd);

  void beginCommandBuffer(VkCommandBuffer cmd);
  void endCommandBuffer(VkCommandBuffer cmd);

  void submit(VkCommandBuffer cmd);
  void waitIdle();

  VkCommandBuffer allocBeginCommandBuffer(VkCommandPool cmdPool);
  void endSubmitWaitCommandBuffer(VkCommandPool cmdPool, VkCommandBuffer cmd);

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
