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

  Buffer createBuffer(std::size_t size, VkBufferUsageFlags usage = 0,
                      VmaAllocationCreateFlags flags = 0);
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

  void copy(VmaAllocation dst, const void *src, std::size_t size) {
    VkResult result = vmaCopyMemoryToAllocation(m_vma, src, dst, 0, size);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to copy memory to allocation.");
    }
  }

  void copy(void *dst, VmaAllocation src, std::size_t size) {
    VkResult result =
        vmaCopyAllocationToMemory(m_vma, src, 0, dst, size);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to copy memory from allocation.");
    }
  }

  void cmdCopy(VkCommandBuffer cmd, Buffer dst, Buffer src, std::size_t size,
               std::size_t dstOffset = 0, std::size_t srcOffset = 0) {
    VkBufferCopy copy;
    copy.size = size;
    copy.srcOffset = srcOffset;
    copy.dstOffset = dstOffset;
    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &copy);
  }

  void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess) {
    VkMemoryBarrier memoryBarrier;
    std::memset(&memoryBarrier, 0, sizeof(VkMemoryBarrier));
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = srcAccess;
    memoryBarrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &memoryBarrier, 0,
                         nullptr, 0, nullptr);
  }

  void cmdBufferBarrier(VkCommandBuffer cmd, Buffer buffer,
                        VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess, VkDeviceSize offset = 0,
                        VkDeviceSize size = VK_WHOLE_SIZE) {
    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = nullptr;
    bufferBarrier.srcAccessMask = srcAccess;
    bufferBarrier.dstAccessMask = dstAccess;
    bufferBarrier.srcQueueFamilyIndex = m_queueFamily;
    bufferBarrier.dstQueueFamilyIndex = m_queueFamily;
    bufferBarrier.buffer = buffer.buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1,
                         &bufferBarrier, 0, nullptr);
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
