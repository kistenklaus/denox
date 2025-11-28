#pragma once
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fmt/printf.h>
#include <span>
#include <stdexcept>
#include <vector>
#include <vma.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

struct Buffer {
  VkBuffer vkbuffer;
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
  void destroyBuffer(const Buffer &buffer);

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

  VkDescriptorPool
  createDescriptorPool(std::size_t maxSets,
                       std::span<const VkDescriptorPoolSize> sizes);

  void destroyDescriptorPool(VkDescriptorPool pool);

  VkDescriptorSet allocDescriptorSet(VkDescriptorPool pool,
                                     VkDescriptorSetLayout layout);

  void allocDescriptorSets(VkDescriptorPool pool,
                           std::span<const VkDescriptorSetLayout> layouts,
                           VkDescriptorSet *sets);

  void updateDescriptorSets(std::span<const VkWriteDescriptorSet> writeInfos);

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

  void copy(VmaAllocation dst, const void *src, std::size_t size);

  void copy(void *dst, VmaAllocation src, std::size_t size);

  void cmdCopy(VkCommandBuffer cmd, Buffer dst, Buffer src, std::size_t size,
               std::size_t dstOffset = 0, std::size_t srcOffset = 0);

  void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess);

  void cmdBufferBarrier(VkCommandBuffer cmd, Buffer buffer,
                        VkPipelineStageFlags srcStage,
                        VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
                        VkAccessFlags dstAccess, VkDeviceSize offset = 0,
                        VkDeviceSize size = VK_WHOLE_SIZE);

  std::uint32_t getQueueFamily() const { return m_queueFamily; }

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

  std::vector<uint64_t> getQueryResults(VkQueryPool queryPool, uint32_t count) {
    std::vector<uint64_t> timestamps(count);
    vkGetQueryPoolResults(m_device, queryPool, 0, count,
                          timestamps.size() * sizeof(uint64_t),
                          timestamps.data(), sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return timestamps;
  }

  float timestampDifference(std::span<const uint64_t> timestamps,
                            uint32_t begin, uint32_t end) {
    uint64_t b = timestamps[begin];
    uint64_t e = timestamps[end];
    return static_cast<float>(e - b) * m_timestampPeriod / 1000000.0f;
  }

  uint64_t timestampNanoDifference(std::span<const uint64_t> timestamps,
                                   uint32_t begin, uint32_t end) {
    uint64_t b = timestamps[begin];
    uint64_t e = timestamps[end];
    return static_cast<uint64_t>(
        std::round(static_cast<float>(e - b) * m_timestampPeriod));
  }

private:
  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  std::uint32_t m_queueFamily;
  VkQueue m_queue;
  VmaAllocator m_vma;
  float m_timestampPeriod;
};

} // namespace denox::runtime
