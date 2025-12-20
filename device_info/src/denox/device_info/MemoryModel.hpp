#pragma once

namespace denox {

struct MemoryModelProperties {
  bool vmm;
  bool vmmDeviceScope;
  bool vmmAvailabilityVisibilityChains;

  bool bufferDeviceAddress;
  bool bufferDeviceAddressCaptureReplay;
  bool bufferDeviceAddressMultiDevice;
};

} // namespace denox::compiler
