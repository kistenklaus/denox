#pragma once

namespace denox::compiler {

struct MemoryModelProperties {
  bool vmm;
  bool vmmDeviceScope;
  bool vmmAvailabilityVisibilityChains;

  bool bufferDeviceAddress;
  bool bufferDeviceAddressCaptureReplay;
  bool bufferDeviceAddressMultiDevice;
};

} // namespace denox::compiler
