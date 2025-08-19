#pragma once

#include "vkcnn/common/FilterMode.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
namespace vkcnn {
class CmdOpBuffer;

class CmdOpUpsample {
public:
  friend CmdOpBuffer;
  struct DebugInfo {
    friend CmdOpUpsample;

    std::string_view name() const {
      auto raw = reinterpret_cast<const std::byte *>(this) + sizeof(DebugInfo);
      raw = align_up<alignof(char)>(raw);
      return std::string_view{reinterpret_cast<const char *>(raw), m_nameLen};
    }
    FilterMode mode() const {
      return m_filterMode;
    }

    unsigned int scalingFactor() const {
      return m_scalingFactor;
    }

  private:
    DebugInfo(FilterMode mode, unsigned int scalingFactor, unsigned int nameLen)
        : m_filterMode(mode), m_scalingFactor(scalingFactor),
          m_nameLen(nameLen) {}

    FilterMode m_filterMode;
    unsigned int m_scalingFactor;

    unsigned int m_nameLen;
    // char m_name[]; <- FAM
  };

private:
  CmdOpUpsample(unsigned int inputTensor, unsigned int outputTensor,
                std::uint64_t spirvSrcSize, bool debugInfo)
      : m_inputTensor(inputTensor), m_outputTensor(outputTensor),
        m_spirvSrcSize(debugInfo
                           ? (uint64_t(0x8000000000000000ull) | spirvSrcSize)
                           : (spirvSrcSize)) {}

  void *spirvPtr() {
    std::byte *self = reinterpret_cast<std::byte *>(this);
    std::byte *end = self + sizeof(CmdOpUpsample);
    std::byte *spirvSrc = align_up<alignof(std::byte)>(end);
    return spirvSrc;
  }

  const void *spirvPtr() const {
    const std::byte *self = reinterpret_cast<const std::byte *>(this);
    const std::byte *end = self + sizeof(CmdOpUpsample);
    const std::byte *spirvSrc = align_up<alignof(std::uint32_t)>(end);
    return spirvSrc;
  }

  std::uint64_t byteSize(bool includeDebugInfo) const {
    std::uint64_t size = sizeof(CmdOpUpsample);
    size = align_up<alignof(std::uint32_t)>(size);
    size += spirvSrcSize() * sizeof(std::uint32_t);
    if (includeDebugInfo) {
      size = align_up<alignof(DebugInfo)>(size);
      auto raw = reinterpret_cast<const std::byte *>(this) + size;
      std::uint64_t nameLen =
          reinterpret_cast<const DebugInfo *>(raw)->m_nameLen;

      size = align_up<alignof(char)>(size + sizeof(DebugInfo));
      size += nameLen * sizeof(char);
    }
    return size;
  }

  static void construct(CmdOpUpsample *ptr, unsigned int inputTensor,
                        unsigned int outputTensor,
                        std::span<const std::uint32_t> spirv,
                        bool debugInfo = false) {
    assert(spirv.size_bytes() % sizeof(std::uint32_t));
    new (ptr) CmdOpUpsample(inputTensor, outputTensor, spirv.size(), debugInfo);
    std::memcpy(ptr->spirvPtr(), spirv.data(), spirv.size_bytes());
  }

  static void construct(CmdOpUpsample *ptr, unsigned int inputTensor,
                        unsigned int outputTensor,
                        std::span<const std::uint32_t> spirv, FilterMode mode,
                        unsigned int scalingFactor, std::string_view name,
                        bool includeDebugInfo) {
    construct(ptr, inputTensor, outputTensor, spirv, includeDebugInfo);
    if (includeDebugInfo) {
      auto raw = static_cast<std::byte *>(ptr->spirvPtr()) + spirv.size_bytes();
      raw = align_up<alignof(DebugInfo)>(raw);
      new (raw) DebugInfo(mode, scalingFactor, name.size());
      raw = align_up<alignof(std::string_view::value_type)>(raw +
                                                            sizeof(DebugInfo));
      std::memcpy(raw, name.data(),
                  name.size() *
                      sizeof(std::basic_string_view<std::uint8_t>::value_type));
    }
  }

  std::uint64_t spirvSrcSize() const {
    return m_spirvSrcSize & uint64_t(0x7FFFFFFFFFFFFFFFull);
  }

  std::optional<DebugInfo> debugInfo() const {
    if ((m_spirvSrcSize & uint64_t(0x8000000000000000ull)) != 0) {
      auto raw = static_cast<const std::byte *>(spirvPtr());
      raw = align_up<alignof(DebugInfo)>(raw);
      return *reinterpret_cast<const DebugInfo *>(raw);
    } else {
      return std::nullopt;
    }
  }

  unsigned int m_inputTensor;
  unsigned int m_outputTensor;
  std::uint64_t m_spirvSrcSize;
  // std::uint32_t m_spirvSrc[]; <- FAM
};

} // namespace vkcnn
