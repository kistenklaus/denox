#pragma once

#include "denox/device_info/DeviceInfo.hpp"
#include <span>

denox::DeviceInfo deserialize_device_yml(std::span<const std::byte> bytes);

denox::memory::vector<std::byte>
serialize_device_yml(const denox::DeviceInfo &deviceInfo);

