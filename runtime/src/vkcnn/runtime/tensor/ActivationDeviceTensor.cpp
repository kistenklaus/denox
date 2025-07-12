#include "./ActivationDeviceTensor.hpp"
#include <stdexcept>

namespace vkcnn::runtime {

void ActivationDeviceTensor::Download::complete(
    ActivationHostTensorView hostTensor) {
  assert(hostTensor.desc() == m_store->desc);
  std::memcpy(hostTensor.data(), m_memory->map(), hostTensor.byteSize());
  m_memory->unmap();
  m_memory = nullptr;
}
void ActivationDeviceTensor::upload(
    const ::merian::CommandBufferHandle &cmd,
    ActivationHostTensorConstView tensor) const {
  assert(m_store->desc == tensor.desc());
  if (m_store->stageManager == nullptr) {
    throw std::runtime_error("Trying to upload to a ActivationDeviceTensor "
                             "without a staging buffer");
  }
  use(cmd, SyncUseFlagBits::TransferWrite);
  m_store->stageManager->cmd_to_device(cmd, m_store->buffer, tensor.data());
}
const ::merian::BufferHandle &
ActivationDeviceTensor::use(const ::merian::CommandBufferHandle &cmd,
                            SyncUseFlags useFlags) const {
  // Only RAR hazards don't require a barrier.
  // If store-sync is None we never require a barrier.
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
ActivationDeviceTensor::Download ActivationDeviceTensor::download(
    const ::merian::CommandBufferHandle &cmd) const {
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
} // namespace vkcnn::runtime
