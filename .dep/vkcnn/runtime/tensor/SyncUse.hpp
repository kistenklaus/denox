#pragma once

#include "vulkan/vulkan.hpp"

namespace vkcnn::runtime {

enum class SyncUseFlagBits {
  None = 0,
  ComputeRead = 1,
  ComputeWrite = 2,
  TransferRead = 4,
  TransferWrite = 8,

  AnyRead = ComputeRead | TransferRead,
  AnyWrite = ComputeWrite | TransferWrite,
};
// Enable bitwise ops for SyncUseBits
inline constexpr SyncUseFlagBits operator|(SyncUseFlagBits a,
                                           SyncUseFlagBits b) {
  return static_cast<SyncUseFlagBits>(static_cast<unsigned>(a) |
                                      static_cast<unsigned>(b));
}

inline constexpr SyncUseFlagBits operator&(SyncUseFlagBits a,
                                           SyncUseFlagBits b) {
  return static_cast<SyncUseFlagBits>(static_cast<unsigned>(a) &
                                      static_cast<unsigned>(b));
}

inline constexpr SyncUseFlagBits operator^(SyncUseFlagBits a,
                                           SyncUseFlagBits b) {
  return static_cast<SyncUseFlagBits>(static_cast<unsigned>(a) ^
                                      static_cast<unsigned>(b));
}

inline constexpr SyncUseFlagBits operator~(SyncUseFlagBits a) {
  return static_cast<SyncUseFlagBits>(~static_cast<unsigned>(a));
}

inline constexpr bool any(SyncUseFlagBits b) {
  return b != SyncUseFlagBits::None;
}

// Strongly typed wrapper
struct SyncUseFlags {
  constexpr SyncUseFlags() : m_bits(SyncUseFlagBits::None) {}
  constexpr SyncUseFlags(SyncUseFlagBits bits) : m_bits(bits) {}

  constexpr SyncUseFlags &operator|=(SyncUseFlagBits bits) {
    m_bits = m_bits | bits;
    return *this;
  }

  constexpr SyncUseFlags &operator|=(SyncUseFlags flags) {
    m_bits = m_bits | flags.m_bits;
    return *this;
  }

  constexpr SyncUseFlags &operator&=(SyncUseFlagBits bits) {
    m_bits = m_bits & bits;
    return *this;
  }

  constexpr SyncUseFlags &operator&=(SyncUseFlags flags) {
    m_bits = m_bits & flags.m_bits;
    return *this;
  }

  constexpr bool contains(SyncUseFlagBits bits) const {
    return any(m_bits & bits);
  }

  constexpr explicit operator bool() const {
    return m_bits != SyncUseFlagBits::None;
  }

  constexpr SyncUseFlagBits bits() const { return m_bits; }

private:
  SyncUseFlagBits m_bits;
};

inline constexpr SyncUseFlags operator|(SyncUseFlags lhs, SyncUseFlagBits rhs) {
  lhs |= rhs;
  return lhs;
}

inline constexpr SyncUseFlags operator&(SyncUseFlags lhs, SyncUseFlagBits rhs) {
  lhs &= rhs;
  return lhs;
}

static vk::PipelineStageFlags syncToStage(SyncUseFlags flags) {
  vk::PipelineStageFlags stage = {};
  if (flags.contains(SyncUseFlagBits::ComputeRead) ||
      flags.contains(SyncUseFlagBits::ComputeWrite)) {
    stage |= vk::PipelineStageFlagBits::eComputeShader;
  }
  if (flags.contains(SyncUseFlagBits::TransferRead) ||
      flags.contains(SyncUseFlagBits::TransferWrite)) {
    stage |= vk::PipelineStageFlagBits::eTransfer;
  }
  return stage;
}

static vk::AccessFlags syncToAccess(SyncUseFlags flags) {
  vk::AccessFlags access = {};
  if (flags.contains(SyncUseFlagBits::ComputeRead)) {
    access |= vk::AccessFlagBits::eShaderRead;
  }
  if (flags.contains(SyncUseFlagBits::ComputeWrite)) {
    access |= vk::AccessFlagBits::eShaderWrite;
  }
  if (flags.contains(SyncUseFlagBits::TransferRead)) {
    access |= vk::AccessFlagBits::eTransferRead;
  }
  if (flags.contains(SyncUseFlagBits::TransferWrite)) {
    access |= vk::AccessFlagBits::eTransferWrite;
  }
  return access;
}

} // namespace vkcnn::runtime
