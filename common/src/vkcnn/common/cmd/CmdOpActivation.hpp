#pragma once

#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
namespace vkcnn {

class CmdOpBuffer;

// TODO: Refactor this: it's probably significantly better
// to have a shader registry because spirv source sections
// might be duplicated.
// Also we probably should fine a different way to encode debug information,
// possibly with a optional file section or something like this. But then how do
// we store the offsets.
// Maybe with a seperate similarly structured array, where
// only forward iteration is allowed.
// TODO: It would also be nice to have a offset table for random access.
//
// NOTE: This is a FAM (Flexible array layout) struct. Don't try to naivly
// instantiate it!
// The primary reason for FAM is easy and fast serialization!
class CmdOpActivation {
public:
  friend CmdOpBuffer;
  struct DebugInfo {
    friend CmdOpActivation;
    ActivationFunction function() const { return m_func; }

    std::string_view name() const {
      auto raw = reinterpret_cast<const std::byte *>(this) + sizeof(DebugInfo);
      raw = align_up<alignof(char)>(raw);
      return std::string_view{reinterpret_cast<const char *>(raw), m_nameLen};
    }

  private:
    DebugInfo(ActivationFunction func, unsigned int nameLen)
        : m_func(func), m_nameLen(nameLen) {}
    ActivationFunction m_func;
    unsigned int m_nameLen;
    // char name[]; // <- null terminated string.
  };

  std::span<const std::uint32_t> spirv() const {
    return std::span{static_cast<const std::uint32_t *>(spirvPtr()),
                     spirvSrcSize()};
  }

  unsigned int inputTensor() const { return m_inputTensor; }

  unsigned int outputTensor() const { return m_outputTensor; }

private:
  CmdOpActivation(unsigned int inputTensor, unsigned int outputTensor,
                  std::uint64_t spirvSrcSize, bool debugInfo)
      : m_inputTensor(inputTensor), m_outputTensor(outputTensor),
        m_spirvSrcSize(debugInfo
                           ? (uint64_t(0x8000000000000000ull) | spirvSrcSize)
                           : (spirvSrcSize)) {}

  void *spirvPtr() {
    std::byte *self = reinterpret_cast<std::byte *>(this);
    std::byte *end = self + sizeof(CmdOpActivation);
    std::byte *spirvSrc = align_up<alignof(std::byte)>(end);
    return spirvSrc;
  }

  const void *spirvPtr() const {
    const std::byte *self = reinterpret_cast<const std::byte *>(this);
    const std::byte *end = self + sizeof(CmdOpActivation);
    const std::byte *spirvSrc = align_up<alignof(std::uint32_t)>(end);
    return spirvSrc;
  }

  std::uint64_t byteSize(bool includeDebugInfo) const {
    std::uint64_t size = sizeof(CmdOpActivation);
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

  static void construct(CmdOpActivation *ptr, unsigned int inputTensor,
                        unsigned int outputTensor,
                        std::span<const std::uint32_t> spirv,
                        bool debugInfo = false) {
    assert(spirv.size_bytes() % sizeof(std::uint32_t));
    new (ptr)
        CmdOpActivation(inputTensor, outputTensor, spirv.size(), debugInfo);
    std::memcpy(ptr->spirvPtr(), spirv.data(), spirv.size_bytes());
  }

  static void construct(CmdOpActivation *ptr, unsigned int inputTensor,
                        unsigned int outputTensor,
                        std::span<const std::uint32_t> spirv,
                        ActivationFunction func, std::string_view name,
                        bool includeDebugInfo) {
    construct(ptr, inputTensor, outputTensor, spirv, includeDebugInfo);
    if (includeDebugInfo) {
      auto raw = static_cast<std::byte *>(ptr->spirvPtr()) + spirv.size_bytes();
      raw = align_up<alignof(DebugInfo)>(raw);
      new (raw) DebugInfo(func, name.size());
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
  // std::uint32_t m_sprivSrc[];
  // -g => DebugInfo
};

} // namespace vkcnn
