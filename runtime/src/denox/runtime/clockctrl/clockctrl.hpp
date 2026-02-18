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

  // Clock queries (MHz)
  uint32_t gpu_clock() const;
  uint32_t mem_clock() const;

private:
  void *m_impl;
};

} // namespace denox::runtime
