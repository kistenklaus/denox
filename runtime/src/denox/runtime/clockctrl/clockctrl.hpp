#pragma once

#include "denox/runtime/context.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

enum class clock_policy {
  none,   // driver-managed DVFS
  base,   // guaranteed / app clock
  maximum // max supported SM clock
};

class clockctrl {
public:
  // Construction
  explicit clockctrl(const ContextHandle &context);
  ~clockctrl() noexcept;
  clockctrl(const clockctrl &o) = delete;
  clockctrl &operator=(const clockctrl &o) = delete;
  clockctrl(clockctrl &&o) noexcept
      : m_impl(std::exchange(o.m_impl, nullptr)) {}
  clockctrl &operator=(clockctrl &&o) noexcept {
    std::swap(m_impl, o.m_impl);
    return *this;
  }
  bool available() const noexcept {
    return m_impl != nullptr;
  }

  // Clock policies
  void set_clock_policy(clock_policy policy);
  clock_policy current_policy() const;

  // Clock queries (MHz)
  uint32_t sm_clock_current() const;
  uint32_t sm_clock_base() const;
  uint32_t sm_clock_max() const;

  // Telemetry
  uint32_t temperature_celsius() const;
  uint32_t power_milliwatts() const;
  uint32_t utilization_gpu_percent() const;

  // Identity
  uint32_t device_index() const;
  std::string device_name() const;

private:
  void *m_impl;
};

} // namespace denox::runtime
