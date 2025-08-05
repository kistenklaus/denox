#include "./BiasDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/SyncUse.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <stdexcept>

namespace vkcnn::runtime {

void BiasDeviceTensor::Download::complete(BiasHostTensorView hostTensor) {
  assert(hostTensor.desc() == m_store->desc);
  std::memcpy(hostTensor.data(), m_memory->map(), hostTensor.byteSize());
  m_memory->unmap();
  m_memory = nullptr;
}

void BiasDeviceTensor::upload(const ::merian::CommandBufferHandle &cmd,
                              BiasHostTensorConstView tensor) {
  assert(tensor.desc() == m_store->desc);
  if (m_store->stageManager == nullptr) {
    throw std::runtime_error(
        "Trying to upload to a BiasDeviceTensor without a staging bufferTrying "
        "to upload to a BiasDeviceTensor without a staging buffer.");
  }
  use(cmd, SyncUseFlagBits::TransferWrite);
  m_store->stageManager->cmd_to_device(cmd, m_store->buffer, tensor.data());
}

const ::merian::BufferHandle &
BiasDeviceTensor::use(const ::merian::CommandBufferHandle &cmd,
                      SyncUseFlags useFlags) const {
  bool hazard = m_store->sync && ((useFlags & SyncUseFlagBits::AnyWrite) ||
                                  (m_store->sync & SyncUseFlagBits::AnyWrite));
  if (hazard) {
    vk::PipelineStageFlags firstStage =
        static_cast<vk::PipelineStageFlags>(m_store->sync);
    vk::AccessFlags firstAccess = static_cast<vk::AccessFlags>(m_store->sync);
    vk::PipelineStageFlags secondStage =
        static_cast<vk::PipelineStageFlags>(useFlags);
    vk::AccessFlags secondAccess = static_cast<vk::AccessFlags>(useFlags);
    cmd->barrier(firstStage, secondStage,
                 m_store->buffer->buffer_barrier(firstAccess, secondAccess));
  }
  m_store->sync = useFlags;
  return m_store->buffer;
}

BiasDeviceTensor::Download
BiasDeviceTensor::download(const ::merian::CommandBufferHandle &cmd) {
  if (m_store->stageManager == nullptr) {
    throw std::runtime_error("Trying to download from a ActivationDeviceTensor "
                             "without a staging buffer.");
  }
  use(cmd, SyncUseFlagBits::TransferRead);
  Download download{
      m_store, m_store->stageManager->cmd_from_device(cmd, m_store->buffer)};

  return download;
}

} // namespace vkcnn::runtime
