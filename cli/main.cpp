#include "denox/compiler.hpp"
#include <vector>

int main() {
  denox::DeviceInfo deviceInfo;
  deviceInfo.spirvVersion = {1, 6};
  std::vector<denox::DeviceCoopMatType> coopmatTypes = {
      denox::DeviceCoopMatType{
          .m = 16,
          .k = 16,
          .n = 16,
          .aType = denox::CoopMatElemType::float16_t,
          .bType = denox::CoopMatElemType::float16_t,
          .accType = denox::CoopMatElemType::float16_t,
          .scope = denox::CoopMatScope::Subgroup,
      }};
  deviceInfo.coopmatTypes = coopmatTypes.data();
  deviceInfo.coopmatTypeCount = coopmatTypes.size();
  denox::CompileOptions options;
  options.deviceInfo = &deviceInfo;

  denox::compile("net.onnx", options);
}
