#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "vkcnn/host/DynamicWeightTensor.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>
namespace vkcnn::device {

class WeightTensor {
private:
public:
  WeightTensor() : m_host(), m_device(nullptr) {}
  WeightTensor(const merian::ContextHandle &context, host::DynamicWeightTensor tensor)
      : m_host(tensor.bufferView().begin(), tensor.bufferView().end()) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    m_alloc = resources->resource_allocator();
    m_device =
        m_alloc->createBuffer(m_host.size() * sizeof(std::byte),
                              vk::BufferUsageFlagBits::eTransferDst |
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                              merian::MemoryMappingType::NONE, "weight-tensor");
    m_dirty = true;
  }

  WeightTensor(merian::ContextHandle &context,
               host::DynamicWeightTensor &&tensor)
      : m_host(tensor.detachBuffer()) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    m_alloc = resources->resource_allocator();
    m_device =
        m_alloc->createBuffer(m_host.size() * sizeof(std::byte),
                              vk::BufferUsageFlagBits::eTransferDst |
                                  vk::BufferUsageFlagBits::eStorageBuffer,
                              merian::MemoryMappingType::NONE, "weight-tensor");
    m_dirty = true;
  }

  merian::BufferHandle get() { return m_device; }

  inline bool valid() const { return m_device != nullptr; }
  inline operator bool() const { return valid(); }

  void flush(const merian::CommandBufferHandle &cmd) {
    if (m_dirty) {
      // NOTE: This uses the fact that cmd->copy takes ownership of the stage
      // buffer.
      merian::BufferHandle stage = m_alloc->createBuffer(
          m_host.size() * sizeof(std::byte),
          vk::BufferUsageFlagBits::eTransferSrc,
          merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
          "weight-tensor-stage");
      auto mapped = stage->get_memory()->map_as<std::byte>();
      std::memcpy(mapped, m_host.data(), m_host.size() * sizeof(std::byte));

      cmd->barrier(vk::PipelineStageFlagBits::eHost,
                   vk::PipelineStageFlagBits::eTransfer,
                   stage->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                         vk::AccessFlagBits::eTransferRead));
      vk::BufferCopy copy{0, 0, m_host.size() * sizeof(std::byte)};
      cmd->copy(stage, m_device, copy);
      m_host.clear();
      m_host.shrink_to_fit();

      cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                   vk::PipelineStageFlagBits::eComputeShader,
                   m_device->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eShaderRead));
      m_dirty = false;
    }
  }

  void set(host::DynamicWeightTensor &tensor) {
    m_host.assign(tensor.bufferView().begin(), tensor.bufferView().end());
    m_dirty = true;
  }

private:
  std::vector<std::byte> m_host;
  merian::BufferHandle m_device;
  bool m_dirty;
  merian::ResourceAllocatorHandle m_alloc;
};

} // namespace vkcnn::device
