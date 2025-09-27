#pragma once

namespace denox::compiler {

struct MemoryModelProperties {
  bool vmm;
  bool vmm_deviceScope;
  bool vmm_availablityVisibilityChains;

  bool bufferDeviceAddress;
  bool bufferDeviceAddressCaptureReplay;
  bool bufferDeviceAddressMultiDevice;
};

} // namespace denox::compiler
