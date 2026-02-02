#include "denox/runtime/clockctrl/clockctrl.hpp"
#include "denox/runtime/context.hpp"
#include <vulkan/vulkan_core.h>

#ifdef DENOX_HAS_NVML
#include <mutex>
#include <nvml.h>

namespace denox::runtime {

namespace {
std::once_flag g_nvml_init_once;
bool g_nvml_available = false;
} // namespace

void init_nvml_once() {
  if (nvmlInit_v2() == NVML_SUCCESS) {
    g_nvml_available = true;
  } else {
    g_nvml_available = false;
  }
}

struct NVMLImpl {
  nvmlDevice_t device;
  unsigned int max_sm;
  unsigned int base_sm;
};

clockctrl::clockctrl(const ContextHandle &context) : m_impl(nullptr) {
  // 1. Vulkan extension gate
  if (!context->extPCIbusInfoAvailable()) {
    return;
  }

  // 2. Query Vulkan PCI info
  VkPhysicalDevicePCIBusInfoPropertiesEXT pci{};
  pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

  VkPhysicalDeviceProperties2 props{};
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props.pNext = &pci;

  vkGetPhysicalDeviceProperties2(context->vkPhysicalDevice(), &props);

  // 3. Initialize NVML once
  std::call_once(g_nvml_init_once, init_nvml_once);
  if (!g_nvml_available) {
    return;
  }

  // 4. Enumerate NVML devices and match PCI identity
  unsigned int device_count = 0;
  if (nvmlDeviceGetCount(&device_count) != NVML_SUCCESS) {
    return;
  }

  nvmlDevice_t nvmlDevice = nullptr;
  for (unsigned int i = 0; i < device_count; ++i) {
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(i, &dev) != NVML_SUCCESS)
      continue;

    nvmlPciInfo_t nvml_pci{};
    if (nvmlDeviceGetPciInfo(dev, &nvml_pci) != NVML_SUCCESS)
      continue;

    if (nvml_pci.domain == pci.pciDomain && nvml_pci.bus == pci.pciBus &&
        nvml_pci.device == pci.pciDevice) {
      nvmlDevice = dev;
      break;
    }
  }
  if (nvmlDevice == nullptr) {
    return;
  }
  unsigned int max_sm;
  nvmlDeviceGetMaxClockInfo(nvmlDevice, NVML_CLOCK_SM, &max_sm);

  NVMLImpl *impl = new NVMLImpl{
      .device = nvmlDevice,
      .max_sm = max_sm,
      .base_sm = 0, // TODO!
  };
  m_impl = static_cast<void *>(impl);
}
clockctrl::~clockctrl() noexcept {
  if (m_impl != nullptr) {
    NVMLImpl *impl = static_cast<NVMLImpl *>(m_impl);
    delete impl;
    m_impl = nullptr;
  }
}

#endif

} // namespace denox::runtime
