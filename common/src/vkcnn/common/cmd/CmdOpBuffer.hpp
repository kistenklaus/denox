#pragma once

#include "vkcnn/common/cmd/CmdOp.hpp"
#include "vkcnn/common/cmd/CmdOpBarrier.hpp"
#include "vkcnn/common/cmd/CmdOpDispatch.hpp"
#include "vkcnn/common/cmd/record/CmdRecordBarrier.hpp"
#include "vkcnn/common/cmd/record/CmdRecordDispatch.hpp"
#include "vkcnn/common/cmd/record/SizedRecordHeader.hpp"
#include "vkcnn/common/memory/monotone_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <spdlog/common.h>
#include <vector>

namespace vkcnn {

class CmdOpSchedule;

// Monotone builder pattern for a CmdOpSchedule
class CmdOpBuffer {
public:
  friend CmdOpSchedule;
  CmdOpBuffer() {}

  void dispatch(unsigned int inputTensor, unsigned int outputTensor,
                std::span<const std::uint32_t> spirv,
                std::span<std::span<const std::byte>> parameters) {
    static constexpr std::size_t MAX_PARAMETERS = 8;
    assert(parameters.size() < MAX_PARAMETERS);

    m_op.push_back(CmdOp::Dispatch);
    std::uint64_t spirvOffset = emplaceSpirvSrc(spirv);

    std::array<std::uint64_t, MAX_PARAMETERS> parameterOffsets;
    for (std::size_t p = 0; p < parameters.size(); ++p) {
      parameterOffsets[p] =
          m_paramBuf.allocate(parameters.size_bytes(), alignof(double));
    }

    {
      std::uint64_t offset =
          m_cmdBuf.allocate(sizeof(CmdOpDispatch), alignof(CmdOpDispatch));
      void *ptr = m_cmdBuf.get(offset);
      new (ptr) CmdRecordDispatch(inputTensor, outputTensor, spirvOffset,
                                  parameters.size());
      std::uint64_t paramOffset = m_cmdBuf.allocate(
          parameters.size() * sizeof(std::uint64_t), alignof(std::uint64_t));
      auto paramPtr = static_cast<std::uint64_t *>(m_cmdBuf.get(paramOffset));
      for (std::size_t p = 0; p < parameters.size(); ++p) {
        paramPtr[p] = parameterOffsets[p];
      }
      m_cmdOT.push_back(offset);
    }
  }

  // NOTE: The input tensor is read and the output tensor is written to.
  void barrier(unsigned int inputTensor, unsigned int outputTensor) {
    m_op.push_back(CmdOp::Barrier);

    {
      std::uint64_t offset =
          m_cmdBuf.allocate(sizeof(CmdOpBarrier), alignof(CmdOpBarrier));
      void *ptr = m_cmdBuf.get(offset);
      new (ptr) CmdRecordBarrier(inputTensor, outputTensor);
      m_cmdOT.push_back(offset);
    }
  }

private:
  struct HashedSpanKey {
    std::uint64_t hash;
    std::uint64_t offset;
    std::uint64_t size;
  };
  template <class T> inline void hash_combine(std::uint64_t &seed, const T &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  template <class T>
  inline void hash_span(std::uint64_t &seed, const std::span<const T> span) {
    std::hash<T> hasher;
    for (const auto &v : span) {
      seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
  }

  std::uint64_t emplaceSpirvSrc(std::span<const std::uint32_t> spirv) {
    std::uint64_t hash = 0x14079109742;
    hash_span(hash, spirv);
    auto it = std::ranges::find_if(m_spirvOT, [&](const HashedSpanKey &k) {
      if (k.hash != hash) {
        return false;
      }
      return std::ranges::equal(
          spirv,
          std::span<const std::uint32_t>{
              reinterpret_cast<const std::uint32_t *>(m_spirvBuf.get(k.offset)),
              k.size});
    });
    if (it != m_spirvOT.end()) {
      return it->offset;
    } else {
      std::uint64_t headerOffset = m_spirvBuf.allocate(
          sizeof(SizedRecordHeader), alignof(SizedRecordHeader));
      std::uint64_t srcOffset =
          m_spirvBuf.allocate(spirv.size_bytes(), alignof(std::uint32_t));
      // NOTE: If we know headerOffset the srcOffset is always implied!!

      void *headerPtr = m_spirvBuf.get(headerOffset);
      new (headerPtr) SizedRecordHeader(spirv.size());

      void *srcPtr = m_spirvBuf.get(srcOffset);
      std::memcpy(srcPtr, spirv.data(), spirv.size_bytes());
      m_spirvOT.emplace_back(hash, headerOffset, spirv.size());
      return headerOffset;
    }
  }

  std::uint64_t emplaceParam(std::span<const std::byte> weights) {
    std::uint64_t hash = 0x14079109742;
    hash_span(hash, weights);
    auto it = std::ranges::find_if(m_paramOT, [&](const HashedSpanKey &k) {
      if (k.hash != hash) {
        return false;
      }
      return std::ranges::equal(
          weights,
          std::span<const std::byte>{
              reinterpret_cast<const std::byte *>(m_paramBuf.get(k.offset)),
              k.size});
    });
    if (it != m_paramOT.end()) {
      return it->offset;
    } else {
      std::uint64_t headerOffset = m_paramBuf.allocate(
          sizeof(SizedRecordHeader), alignof(SizedRecordHeader));
      std::uint64_t srcOffset =
          m_spirvBuf.allocate(weights.size_bytes(), alignof(double));
      // NOTE: If we know headerOffset the srcOffset is always implied!!

      void *headerPtr = m_spirvBuf.get(headerOffset);
      new (headerPtr) SizedRecordHeader(weights.size_bytes());

      void *srcPtr = m_spirvBuf.get(srcOffset);
      std::memcpy(srcPtr, weights.data(), weights.size_bytes());
      m_spirvOT.emplace_back(hash, headerOffset, weights.size_bytes());
      return headerOffset;
    }
  }

  std::vector<CmdOp> m_op;

  container::monotone_buffer m_cmdBuf;
  std::vector<std::uint64_t> m_cmdOT;

  container::monotone_buffer m_spirvBuf;
  container::monotone_buffer m_paramBuf;

  // Builder only! (Not part of the serialization)
  std::vector<HashedSpanKey> m_spirvOT;
  std::vector<HashedSpanKey> m_paramOT;
};

} // namespace vkcnn
