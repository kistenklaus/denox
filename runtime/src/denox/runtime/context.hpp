#pragma once
#include "denox/device_info/ApiVersion.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include "vma.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fmt/printf.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

struct Buffer {
  VkBuffer vkbuffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
};

class Context {
public:
  static std::shared_ptr<Context> make(const char *deviceName,
                                       ApiVersion apiVersion) {
    return std::shared_ptr<Context>(new Context(deviceName, apiVersion));
  }

  ~Context();
  Context(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(const Context &) = delete;
  Context &operator=(Context &&) = delete;

  Buffer createBuffer(size_t size, VkBufferUsageFlags usage,
                      VmaAllocationCreateFlags flags = 0);
  void destroyBuffer(const Buffer &buffer);

  VkDescriptorSetLayout createDescriptorSetLayout(
      memory::span<const VkDescriptorSetLayoutBinding> bindings);
  void destroyDescriptorSetLayout(VkDescriptorSetLayout layout);

  VkPipelineLayout createPipelineLayout(
      memory::span<const VkDescriptorSetLayout> descriptorLayouts,
      uint32_t pushConstantRange);
  void destroyPipelineLayout(VkPipelineLayout layout);

  VkPipeline createComputePipeline(VkPipelineLayout layout,
                                   memory::span<const uint32_t> binary,
                                   const char *entry);
  void destroyPipeline(VkPipeline pipeline);

  VkDescriptorPool
  createDescriptorPool(uint32_t maxSets,
                       memory::span<const VkDescriptorPoolSize> sizes);

  void destroyDescriptorPool(VkDescriptorPool pool);

  VkDescriptorSet allocDescriptorSet(VkDescriptorPool pool,
                                     VkDescriptorSetLayout layout);

  void allocDescriptorSets(VkDescriptorPool pool,
                           memory::span<const VkDescriptorSetLayout> layouts,
                           VkDescriptorSet *sets);

  void
  updateDescriptorSets(memory::span<const VkWriteDescriptorSet> writeInfos);

  VkCommandPool createCommandPool();
  void destroyCommandPool(VkCommandPool cmdPool);

  VkCommandBuffer allocCommandBuffer(VkCommandPool cmdPool);
  void freeCommandBuffer(VkCommandPool cmdPool, VkCommandBuffer cmd);

  void beginCommandBuffer(VkCommandBuffer cmd);
  void resetCommandBuffer(VkCommandBuffer cmd) {
    VkResult result = vkResetCommandBuffer(cmd, 0);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to reset command buffer!");
    }
  }
  void endCommandBuffer(VkCommandBuffer cmd);

  void submit(VkCommandBuffer cmd);
  void waitIdle();

  VkCommandBuffer allocBeginCommandBuffer(VkCommandPool cmdPool);
  void endSubmitWaitCommandBuffer(VkCommandPool cmdPool, VkCommandBuffer cmd);

  void copy(VmaAllocation dst, const void *src, size_t size);

  void copy(void *dst, VmaAllocation src, size_t size, size_t srcOffset = 0);

  void cmdCopy(VkCommandBuffer cmd, Buffer dst, Buffer src, size_t size,
               size_t dstOffset = 0, size_t srcOffset = 0);

  void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess);

  void cmdMemoryBarrierComputeShader(VkCommandBuffer cmd);

  void cmdBufferBarrier(VkCommandBuffer cmd, Buffer buffer,
                        VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess, VkDeviceSize offset = 0,
                        VkDeviceSize size = VK_WHOLE_SIZE);

  uint32_t getQueueFamily() const { return m_queueFamily; }

  VkQueryPool createTimestampQueryPool(uint32_t timestampCount) {
    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = timestampCount;
    VkQueryPool queryPool;
    VkResult result =
        vkCreateQueryPool(m_device, &createInfo, nullptr, &queryPool);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create timestamp query pool");
    }
    return queryPool;
  }

  void destroyQueryPool(VkQueryPool queryPool) {
    vkDestroyQueryPool(m_device, queryPool, nullptr);
  }

  void cmdResetQueryPool(VkCommandBuffer cmd, VkQueryPool queryPool,
                         uint32_t first, uint32_t queryCount) {
    vkCmdResetQueryPool(cmd, queryPool, first, queryCount);
  }

  void cmdWriteTimestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage,
                         VkQueryPool queryPool, uint32_t query) {
    vkCmdWriteTimestamp(cmd, stage, queryPool, query);
  }

  memory::vector<uint64_t> getQueryResults(VkQueryPool queryPool,
                                           uint32_t count) {
    memory::vector<uint64_t> timestamps(count);
    vkGetQueryPoolResults(m_device, queryPool, 0, count,
                          timestamps.size() * sizeof(uint64_t),
                          timestamps.data(), sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return timestamps;
  }

  float timestampDifference(memory::span<const uint64_t> timestamps,
                            uint32_t begin, uint32_t end) {
    uint64_t b = timestamps[begin];
    uint64_t e = timestamps[end];
    return static_cast<float>(e - b) * m_timestampPeriod / 1000000.0f;
  }

  uint64_t timestampNanoDifference(memory::span<const uint64_t> timestamps,
                                   uint32_t begin, uint32_t end) {
    uint64_t b = timestamps[begin];
    uint64_t e = timestamps[end];
    return static_cast<uint64_t>(
        std::round(static_cast<float>(e - b) * m_timestampPeriod));
  }

  VkFence createFence(bool signaled) {
    VkFenceCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    createInfo.pNext = nullptr;
    if (signaled) {
      createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    } else {
      createInfo.flags = 0;
    }
    VkFence fence;
    {
      VkResult result = vkCreateFence(m_device, &createInfo, nullptr, &fence);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence.");
      }
    }
    return fence;
  }

  void waitFence(VkFence fence) {
    VkResult result = vkWaitForFences(m_device, 1, &fence, VK_FALSE,
                                      std::numeric_limits<uint64_t>::max());
    if (result == VK_TIMEOUT) {
      std::cerr << fmt::format("vkWaitForFences timed out") << std::endl;
    }
    if (result == VK_ERROR_DEVICE_LOST) {
      throw std::runtime_error("Lost device!");
    }
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to wait for fence or timed out.");
    }
  }

  void resetFence(VkFence fence) {
    VkResult result = vkResetFences(m_device, 1, &fence);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to reset fence.");
    }
  }
  void destroyFence(VkFence fence) { vkDestroyFence(m_device, fence, nullptr); }

  void submit(VkCommandBuffer cmd, VkFence fence) {
    VkSubmitInfo submitInfo;
    std::memset(&submitInfo, 0, sizeof(VkSubmitInfo));
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.pSignalSemaphores = nullptr;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitDstStageMask = nullptr;

    {
      VkResult result = vkQueueSubmit(m_queue, 1, &submitInfo, fence);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit to queue.");
      }
    }
  }

  void resetDescriptorPool(VkDescriptorPool pool) {
    VkResult result = vkResetDescriptorPool(m_device, pool, 0);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to reset descriptor pool");
    }
  }

  VkInstance vkInstance() const { return m_instance; }
  VkPhysicalDevice vkPhysicalDevice() const { return m_physicalDevice; }

  bool extPCIbusInfoAvailable() const { return m_extPCIbufInfoAvailable; }

private:
  explicit Context(const char *deviceName, ApiVersion target_env);

private:
  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  uint32_t m_queueFamily;
  VkQueue m_queue;
  VmaAllocator m_vma;
  float m_timestampPeriod;
  bool m_extPCIbufInfoAvailable;
};

using ContextHandle = std::shared_ptr<Context>;

} // namespace denox::runtime
