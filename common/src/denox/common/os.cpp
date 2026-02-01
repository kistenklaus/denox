// CHATGPT helper hopefully it works reliably,
// can't test on platforms other than linux right now.

#include "denox/common/os.hpp"

#include <fmt/format.h>
#include <string>

#if defined(_WIN32)

#define NOMINMAX
#include <windows.h>

// Undocumented but stable; used by Chromium, LLVM, etc.
extern "C" LONG WINAPI RtlGetVersion(PRTL_OSVERSIONINFOW);

#elif defined(__APPLE__)

#include <sys/sysctl.h>

#elif defined(__linux__)

#include <fstream>
#include <unordered_map>

#endif

namespace denox {

std::string os_name() {

#if defined(_WIN32)

  RTL_OSVERSIONINFOW info{};
  info.dwOSVersionInfoSize = sizeof(info);

  if (RtlGetVersion(&info) == 0) {
    return fmt::format("Windows {}.{}.{}", info.dwMajorVersion,
                       info.dwMinorVersion, info.dwBuildNumber);
  }

  return "Windows (unknown version)";

#elif defined(__APPLE__)

  char version[256];
  size_t size = sizeof(version);

  if (sysctlbyname("kern.osproductversion", version, &size, nullptr, 0) == 0) {
    return fmt::format("macOS {}", version);
  }

  return "macOS (unknown version)";

#elif defined(__linux__)

  std::ifstream file("/etc/os-release");
  if (!file.is_open())
    return "Linux (unknown distribution)";

  std::unordered_map<std::string, std::string> fields;
  std::string line;

  while (std::getline(file, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;

    std::string key = line.substr(0, eq);
    std::string val = line.substr(eq + 1);

    if (!val.empty() && val.front() == '"' && val.back() == '"')
      val = val.substr(1, val.size() - 2);

    fields[key] = val;
  }

  if (fields.contains("NAME") && fields.contains("VERSION")) {
    return fmt::format("{} {}", fields["NAME"], fields["VERSION"]);
  }

  if (fields.contains("PRETTY_NAME")) {
    return fields["PRETTY_NAME"];
  }

  return "Linux (unknown distribution)";

#else

  return "Unknown OS";

#endif
}

} // namespace denox
