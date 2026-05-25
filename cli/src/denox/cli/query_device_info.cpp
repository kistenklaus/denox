#include "query_device_info.hpp"
#include "denox/cli/device_yml/device_yml.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/io/fs/Path.hpp"
#include <span>
#include <yaml-cpp/yaml.h>

void query_device_info(QueryDeviceInfo &action) {

  auto apiVersion = denox::ApiVersion::VULKAN_1_4;

  denox::DeviceInfo deviceInfo =
      denox::query_driver_device_info(apiVersion, action.deviceName);

  denox::memory::vector<std::byte> device_yml =
      serialize_device_yml(deviceInfo);

  IOEndpoint output = [&]() {
    if (action.output.has_value()) {
      return *action.output;
    } else {
      return IOEndpoint{
          denox::io::Path{deviceInfo.name}.with_extension("yaml")};
    }
  }();

  OutputStream outstream{output};
  outstream.write_exact(device_yml);
}
