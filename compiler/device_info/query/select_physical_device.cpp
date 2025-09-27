#include "device_info/query/select_physical_device.hpp"
#include "diag/logging.hpp"
#include <exception>

namespace denox::compiler::device_info::query {

// ASCII lowercase
static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Simple case-insensitive glob: '*' and '?' only.
static bool glob_match_ci(std::string pat, std::string text) {
  pat = to_lower(std::move(pat));
  text = to_lower(std::move(text));

  size_t p = 0, t = 0, star = std::string::npos, match = 0;
  while (t < text.size()) {
    if (p < pat.size() && (pat[p] == '?' || pat[p] == text[t])) {
      ++p;
      ++t;
    } else if (p < pat.size() && pat[p] == '*') {
      star = p++;
      match = t; // remember where '*' is and the match start
    } else if (star != std::string::npos) {
      p = star + 1;
      t = ++match; // backtrack: let '*' eat one more char
    } else {
      return false;
    }
  }
  while (p < pat.size() && pat[p] == '*')
    ++p;
  return p == pat.size();
}

vk::PhysicalDevice select_physical_device(vk::Instance instance,
                                          const memory::string &deviceName) {
  std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
  if (devices.empty()) {
    DENOX_ERROR("Failed to select physical device: No Vulkan device found.");
    std::terminate();
  }

  // Empty name => pick a discrete GPU (first one). Throw if none found.
  if (deviceName.empty()) {
    for (auto &d : devices) {
      auto props = d.getProperties();
      if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        return d;
      }
    }
    return devices[0];
  }

  // Non-empty => treat as glob pattern
  std::vector<vk::PhysicalDevice> matches;
  for (auto &d : devices) {
    auto props = d.getProperties();
    if (glob_match_ci(deviceName.c_str(), props.deviceName)) {
      matches.push_back(d);
    }
  }

  if (matches.empty()) {
    DENOX_ERROR("Failed to select physical device: pattern: \"{}\" did not "
                "match any device.",
                deviceName);
    std::terminate();
  }
  if (matches.size() > 1) {
    // Build a short list for the error message
    std::string list;
    for (auto &d : matches) {
      list += std::string(d.getProperties().deviceName) + "; ";
    }
    DENOX_ERROR("Failed to select physical device: pattern :\"{}\" is "
                "ambiguous, devices: {}",
                deviceName, list);
    std::terminate();
  }

  return matches.front();
}

} // namespace denox::compiler::device_info::query
