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

  constexpr explicit operator bool() const {
    return m_bits != SyncUseFlagBits::None;
  }

  constexpr explicit operator vk::PipelineStageFlags() const {
    vk::PipelineStageFlags stage = {};
    if ((m_bits & SyncUseFlagBits::ComputeRead) != SyncUseFlagBits::None ||
        (m_bits & SyncUseFlagBits::ComputeWrite) != SyncUseFlagBits::None) {
      stage |= vk::PipelineStageFlagBits::eComputeShader;
    }
    if ((m_bits & SyncUseFlagBits::TransferRead) != SyncUseFlagBits::None ||
        (m_bits & SyncUseFlagBits::TransferWrite) != SyncUseFlagBits::None) {
      stage |= vk::PipelineStageFlagBits::eTransfer;
    }
    return stage;
  }
  constexpr explicit operator vk::AccessFlags() const {
    vk::AccessFlags access = {};
    if ((m_bits & SyncUseFlagBits::ComputeRead) != SyncUseFlagBits::None) {
      access |= vk::AccessFlagBits::eShaderRead;
    }
    if ((m_bits & SyncUseFlagBits::ComputeWrite) != SyncUseFlagBits::None) {
      access |= vk::AccessFlagBits::eShaderWrite;
    }
    if ((m_bits & SyncUseFlagBits::TransferRead) != SyncUseFlagBits::None) {
      access |= vk::AccessFlagBits::eTransferRead;
    }
    if ((m_bits & SyncUseFlagBits::TransferWrite) != SyncUseFlagBits::None) {
      access |= vk::AccessFlagBits::eTransferWrite;
    }
    return access;
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

} // namespace vkcnn::runtime
