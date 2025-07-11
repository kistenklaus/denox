#pragma once

#include <cstdint>
#include <print>
#include <span>
#include <vector>

namespace vkcnn {

class SpecializationConstants {
public:
  SpecializationConstants(std::vector<std::uint32_t> constants)
      : m_constants(std::move(constants)) {}

  SpecializationConstants(std::span<const std::uint32_t> constants = {})
      : m_constants(constants.begin(), constants.end()) {
      }

  auto begin() { return m_constants.begin(); }
  auto end() { return m_constants.end(); }
  auto begin() const { return m_constants.begin(); }
  auto end() const { return m_constants.end(); }

private:
  std::vector<std::uint32_t> m_constants;
};

} // namespace vkcnn
