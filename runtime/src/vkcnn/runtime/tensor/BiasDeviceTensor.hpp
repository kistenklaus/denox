#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "vkcnn/common/tensor/BiasDescriptor.hpp"
#include "vkcnn/common/tensor/BiasHostTensor.hpp"
#include "vkcnn/runtime/tensor/SyncUse.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <concepts>
#include <fmt/base.h>
namespace vkcnn::runtime {

class BiasDeviceTensor {
  struct Storage {
    BiasDescriptor desc;
    ::merian::BufferHandle buffer;
    ::merian::StagingMemoryManagerHandle stageManager;
    SyncUseFlags sync = SyncUseFlagBits::None;
  };

public:
  class Download {
  public:
    friend class BiasDeviceTensor;
    void complete(BiasHostTensorView hostTensor);

    template <typename Alloc = std::allocator<std::byte>>
      requires(!std::same_as<Alloc, BiasHostTensorView> &&
               !std::same_as<Alloc, BiasHostTensor>)
    BiasHostTensor complete(const Alloc &alloc = {}) {
      BiasHostTensor tensor{m_store->desc, alloc};
      complete(tensor);
      return tensor;
    }

  private:
    Download(std::shared_ptr<Storage> store,
             ::merian::MemoryAllocationHandle memory)
        : m_store(std::move(store)), m_memory(std::move(memory)) {}

    std::shared_ptr<Storage> m_store;
    ::merian::MemoryAllocationHandle m_memory;
  };

  template <typename Alloc = std::allocator<std::byte>>
  BiasDeviceTensor(BiasDescriptor desc,
                   const ::merian::ResourceAllocatorHandle &deviceAlloc,
                   bool disableStage = false, const Alloc &alloc = {})
      : m_store(std::allocate_shared<Storage>(alloc, desc, nullptr, nullptr)) {
    const std::size_t byteSize = m_store->desc.byteSize();
    if (disableStage) {
      m_store->buffer = deviceAlloc->createBuffer(
          byteSize, vk::BufferUsageFlagBits::eStorageBuffer,
          ::merian::MemoryMappingType::NONE);
      m_store->stageManager = nullptr;
    } else {
      m_store->buffer =
          deviceAlloc->createBuffer(byteSize,
                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                        vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    ::merian::MemoryMappingType::NONE);
      m_store->stageManager = deviceAlloc->getStaging();
    }
  }

  void upload(const ::merian::CommandBufferHandle &cmd,
              BiasHostTensorConstView tensor);

  const ::merian::BufferHandle &buffer() const { return m_store->buffer; }
  const ::merian::BufferHandle &use(const ::merian::CommandBufferHandle &cmd,
                                    SyncUseFlags useFlags) const;

  Download download(const ::merian::CommandBufferHandle &cmd);

  const BiasDescriptor &desc() const { return m_store->desc; }
  unsigned int shape() const { return m_store->desc.shape; }
  FloatType type() const { return m_store->desc.type; }

private:
  std::shared_ptr<Storage> m_store;
};

} // namespace vkcnn::runtime
