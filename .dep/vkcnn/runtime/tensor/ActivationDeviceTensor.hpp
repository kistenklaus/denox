#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/runtime/tensor/SyncUse.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <concepts>
#include <iterator>
#include <memory>
#include <stdexcept>
namespace vkcnn::runtime {

class ActivationDeviceTensor {
public:
  struct Storage {
    ActivationDescriptor desc;
    merian::BufferHandle buffer;
    merian::StagingMemoryManagerHandle stageManager;
    SyncUseFlags sync = SyncUseFlagBits::None;
  };

public:
  class Download {
  public:
    friend class ActivationDeviceTensor;
    void complete(ActivationHostTensorView hostTensor) {
      assert(hostTensor.desc() == m_store->desc);
      std::memcpy(hostTensor.data(), m_memory->map(), hostTensor.byteSize());
      m_memory->unmap();
      m_memory = nullptr;
    }

    template <typename Alloc = std::allocator<std::byte>>
      requires(!std::same_as<Alloc, ActivationHostTensorView> &&
               !std::same_as<Alloc, ActivationHostTensor>)
    ActivationHostTensor complete(const Alloc &alloc = {}) {
      ActivationHostTensor tensor{m_store->desc, alloc};
      complete(tensor);
      return tensor;
    }

  private:
    Download(std::shared_ptr<Storage> store,
             merian::MemoryAllocationHandle memory)
        : m_store(store), m_memory(memory) {}
    std::shared_ptr<Storage> m_store;
    merian::MemoryAllocationHandle m_memory;
  };
  template <typename Alloc = std::allocator<std::byte>>
  ActivationDeviceTensor(ActivationDescriptor desc,
                         const merian::ResourceAllocatorHandle &deviceAlloc,
                         bool disableStage = false, const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(alloc, desc, nullptr, nullptr)) {
    const std::size_t byteSize = m_store->desc.byteSize();
    if (disableStage) {
      m_store->buffer = deviceAlloc->createBuffer(
          byteSize, vk::BufferUsageFlagBits::eStorageBuffer,
          merian::MemoryMappingType::NONE);
      m_store->stageManager = nullptr;
    } else {
      m_store->buffer =
          deviceAlloc->createBuffer(byteSize,
                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                        vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    merian::MemoryMappingType::NONE);
      m_store->stageManager = deviceAlloc->getStaging();
    }
  }

  void upload(const merian::CommandBufferHandle &cmd,
              ActivationHostTensorConstView tensor) const {
    assert(m_store->desc == tensor.desc());
    if (m_store->stageManager == nullptr) {
      throw std::runtime_error("Trying to upload to a ActivationDeviceTensor "
                               "without a staging buffer");
    }
    use(cmd, SyncUseFlagBits::TransferWrite);
    m_store->stageManager->cmd_to_device(cmd, m_store->buffer, tensor.data());
  }

  const merian::BufferHandle &buffer() const { return m_store->buffer; }

  const merian::BufferHandle &use(const merian::CommandBufferHandle &cmd,
                                  SyncUseFlags useFlags) const {
    // Only RAR hazards don't require a barrier.
    // If store-sync is None we never require a barrier.
    bool hazard =
        m_store->sync && ((useFlags & SyncUseFlagBits::AnyWrite) ||
                          (m_store->sync & SyncUseFlagBits::AnyWrite));

    if (hazard) {
      vk::PipelineStageFlags firstStage = syncToStage(m_store->sync);
      vk::AccessFlags firstAccess = syncToAccess(m_store->sync);
      vk::PipelineStageFlags secondStage = syncToStage(useFlags);
      vk::AccessFlags secondAccess = syncToAccess(useFlags);
      cmd->barrier(firstStage, secondStage,
                   m_store->buffer->buffer_barrier(firstAccess, secondAccess));
    }
    m_store->sync = useFlags;
    return m_store->buffer;
  }

  Download download(const merian::CommandBufferHandle &cmd) const {
    if (m_store->stageManager == nullptr) {
      throw std::runtime_error(
          "Trying to download from a ActivationDeviceTensor without a staging "
          "buffer");
    }
    use(cmd, SyncUseFlagBits::TransferRead);
    Download download{
        m_store, m_store->stageManager->cmd_from_device(cmd, m_store->buffer)};

    return download;
  }

  unsigned int w() const { return m_store->desc.shape.w; }
  unsigned int h() const { return m_store->desc.shape.h; }
  unsigned int c() const { return m_store->desc.shape.c; }

  ActivationDescriptor desc() const { return m_store->desc; }
  ActivationShape shape() const { return m_store->desc.shape; }
  ActivationLayout layout() const { return m_store->desc.layout; }
  FloatType type() const { return m_store->desc.type; }

private:
  std::shared_ptr<Storage> m_store;
};

} // namespace vkcnn::runtime
