#pragma once
#include "denox/compiler.hpp"
#include "denox/runtime.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace pydenox::details {
struct ContextKey {
  std::optional<std::string> device;
  denox::VulkanApiVersion target_env;

  friend bool operator==(const ContextKey &lhs, const ContextKey &rhs) {
    if (lhs.device != rhs.device) {
      return false;
    }
    if (lhs.target_env != rhs.target_env) {
      return false;
    }
    return true;
  }
};
} // namespace pydenox::details

namespace std {
template <> struct hash<pydenox::details::ContextKey> {
  size_t operator()(const pydenox::details::ContextKey &key) const noexcept {
    size_t h = 0;

    // hash combine helper (FNV-1a inspired)
    auto hash_combine = [&](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };

    // Hash device if present
    if (key.device.has_value())
      hash_combine(std::hash<std::string>{}(key.device.value()));
    else
      hash_combine(0xdeadbeef); // arbitrary salt for nullopt

    // Hash enum using underlying type
    using UnderT = std::underlying_type_t<denox::VulkanApiVersion>;
    hash_combine(std::hash<UnderT>{}(static_cast<UnderT>(key.target_env)));

    return h;
  }
};

} // namespace std

namespace pydenox::details {

class ContextManager {
public:
  ContextManager() {}

  denox::RuntimeContext requireContext(std::optional<std::string> device,
                                       denox::VulkanApiVersion target_env) {
    details::ContextKey key{std::move(device), target_env};
    if (m_contextCache.contains(key)) {
      return m_contextCache.at(key);
    } else {
      const char *deviceName = nullptr;
      if (device.has_value()) {
        deviceName = device->c_str();
      }
      denox::RuntimeContext context;
      if (denox::create_runtime_context(deviceName, &context) < 0) {
        throw std::runtime_error("Failed to create denox runtime context.");
      }
      m_contextCache[key] = context;
      return context;
    }
  }

  ~ContextManager() {
    for (const auto &[key, context] : m_contextCache) {
      denox::destroy_runtime_context(context);
    }
    m_contextCache.clear();
  }

private:
  std::unordered_map<details::ContextKey, denox::RuntimeContext> m_contextCache;
};

} // namespace pydenox::details

namespace pydenox {

struct ContextManager {
public:
  ~ContextManager() {
  }

  ContextManager() {}

  denox::RuntimeContext getContextFor(std::optional<std::string> device,
                                      denox::VulkanApiVersion target_env) {
    require();
    assert(m_shared);
    return m_shared->requireContext(std::move(device), target_env);
  }

private:
  void require() {
    static std::weak_ptr<pydenox::details::ContextManager> manager;
    static std::mutex global_require_lock;
    std::lock_guard lck{global_require_lock};
    if (manager.expired()) {
      m_shared = std::make_shared<pydenox::details::ContextManager>();
      manager = m_shared;
    } else {
      m_shared = manager.lock();
    }
  }

  std::shared_ptr<pydenox::details::ContextManager> m_shared;
};

} // namespace pydenox
