#pragma once

#include "vkcnn/host/WeightTensorLayout.hpp"
#include "vkcnn/host/floatx.hpp"
#include "vkcnn/host/fprec.hpp"
#include <SDL2/SDL_keycode.h>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <fmt/base.h>
#include <functional>
#include <print>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace vkcnn::host {

class DynamicWeightTensor {
public:
  DynamicWeightTensor(WeightTensorLayout layout, FPrec precision, std::size_t k,
                      std::size_t c, std::size_t s, std::size_t r)
      : m_layout(layout), m_precision(precision), m_outputChannels(k),
        m_inputChannels(c), m_kernelWidth(s), m_kernelHeight(r),
        m_storage(m_outputChannels * m_inputChannels * m_kernelWidth *
                  m_kernelHeight * FPrec_Size(m_precision)) {}

  template <typename T = float>
  void set(std::size_t k, std::size_t c, std::size_t s, std::size_t r, T v) {
    DynamicWeightTensor::set<T>(m_storage,
                                getIndexOf(m_layout, k, c, s, r,
                                           m_outputChannels, m_inputChannels,
                                           m_kernelWidth, m_kernelHeight),
                                m_precision, v);
  }

  template <typename T = float>
  T get(std::size_t k, std::size_t c, std::size_t s, std::size_t r) const {
    return DynamicWeightTensor::get<T>(
        m_storage,
        getIndexOf(m_layout, k, c, s, r, m_outputChannels, m_inputChannels,
                   m_kernelWidth, m_kernelHeight),
        m_precision);
  }

  void transformLayout(WeightTensorLayout newLayout) {
    auto oldStorage = m_storage;
    auto oldLayout = m_layout;
    for (std::size_t k = 0; k < m_outputChannels; ++k) {
      for (std::size_t c = 0; c < m_inputChannels; ++c) {
        for (std::size_t s = 0; s < m_kernelWidth; ++s) {
          for (std::size_t r = 0; r < m_kernelHeight; ++r) {
            copy(m_storage,
                 getIndexOf(newLayout, k, c, s, r, m_outputChannels,
                            m_inputChannels, m_kernelWidth, m_kernelHeight),
                 oldStorage,
                 getIndexOf(oldLayout, k, c, s, r, m_outputChannels,
                            m_inputChannels, m_kernelWidth, m_kernelHeight),
                 m_precision);
          }
        }
      }
    }
    m_layout = newLayout;
  }

  void transformPrecision(FPrec newPrecision) {
    if (newPrecision == m_precision) {
      return;
    }
    auto oldStorage = std::move(m_storage);
    auto oldPrec = m_precision;
    auto fSize = FPrec_Size(newPrecision);
    std::size_t size =
        m_outputChannels * m_inputChannels * m_kernelWidth * m_kernelHeight;
    m_storage.resize(size * fSize);
    for (std::size_t i = 0; i < size; ++i) {
      cast(m_storage, i, newPrecision, oldStorage, i, oldPrec);
    }
  }

  template <typename T> void fill(std::function<T(std::size_t)> gen) {
    std::size_t size =
        m_outputChannels * m_inputChannels * m_kernelWidth * m_kernelHeight;
    for (std::size_t i = 0; i < size; ++i) {
      DynamicWeightTensor::set<T>(m_storage, i, m_precision, gen(i));
    }
  }

  inline unsigned int k() const { return m_outputChannels; }
  inline unsigned int c() const { return m_inputChannels; }
  inline unsigned int s() const { return m_kernelWidth; }
  inline unsigned int r() const { return m_kernelHeight; }
  inline FPrec precision() const { return m_precision; }
  inline std::span<const std::byte> bufferView() const { return m_storage; }
  inline std::vector<std::byte> detachBuffer() { return std::move(m_storage); }

private:
  template <typename T>
  __attribute__((always_inline)) static inline T
  get(const std::vector<std::byte> &storage, std::size_t idx, FPrec precision) {
    switch (precision) {
    case FPrec::F16:
      if constexpr (std::same_as<T, f16>) {
        return *reinterpret_cast<const f16 *>(storage.data() + idx * 2);
      } else {
        return static_cast<T>(
            *reinterpret_cast<const f16 *>(storage.data() + idx * 2));
      }
    case FPrec::F32:
      return T(*reinterpret_cast<const f32 *>(storage.data() + idx * 2));
    case FPrec::F64:
      return T(*reinterpret_cast<const f64 *>(storage.data() + idx * 2));
    }
  }

  template <typename T>
  __attribute__((always_inline)) static inline void
  set(std::vector<std::byte> &storage, std::size_t idx, FPrec precision, T v) {
    switch (precision) {
    case FPrec::F16:
      *reinterpret_cast<f16 *>(storage.data() + idx * 2) = f16(v);
      break;
    case FPrec::F32:
      if constexpr (std::same_as<T, f16>) {
        *reinterpret_cast<f32 *>(storage.data() + idx * 2) =
            static_cast<f32>(v);
      } else {
        *reinterpret_cast<f32 *>(storage.data() + idx * 2) = f32(v);
      }
      break;
    case FPrec::F64:
      if constexpr (std::same_as<T, f16>) {
        *reinterpret_cast<f64 *>(storage.data() + idx * 2) =
            static_cast<f64>(v);
      } else {
        *reinterpret_cast<f64 *>(storage.data() + idx * 2) = f64(v);
      }
      break;
    }
  }

  __attribute__((always_inline)) static inline void
  copy(std::vector<std::byte> &dst, std::size_t dstIdx,
       const std::vector<std::byte> &src, std::size_t srcIdx, FPrec precision) {
    std::size_t fsize = FPrec_Size(precision);
    std::memcpy(dst.data() + dstIdx * fsize, src.data() + srcIdx * fsize,
                fsize);
  }

  /// Hopefully this is inlined and the branching is put outside of the loop.
  __attribute__((always_inline)) static inline void
  cast(std::vector<std::byte> &dst, std::size_t dstIdx, FPrec dstPrec,
       const std::vector<std::byte> &src, std::size_t srcIdx, FPrec srcPrec) {
    if (dstPrec == srcPrec) {
      return copy(dst, dstIdx, src, srcIdx, srcPrec);
    }
    switch (dstPrec) {
    case FPrec::F16: {
      f16 *d = reinterpret_cast<f16 *>(dst.data() + 2 * dstIdx);
      switch (srcPrec) {
      case FPrec::F16:
        std::unreachable();
        break;
      case FPrec::F32:
        *d = f16(*reinterpret_cast<const f32 *>((src.data() + 4 * srcIdx)));
        break;
      case FPrec::F64:
        *d = f16(*reinterpret_cast<const f64 *>((src.data() + 8 * srcIdx)));
        break;
      }
    }
    case FPrec::F32: {
      f32 *d = reinterpret_cast<f32 *>(dst.data() + 4 * dstIdx);
      switch (srcPrec) {
      case FPrec::F16:
        *d = static_cast<f32>(
            *reinterpret_cast<const f16 *>(src.data() + 2 * srcIdx));
        break;
      case FPrec::F32:
        std::unreachable();
        break;
      case FPrec::F64:
        *d = f64(*reinterpret_cast<const f64 *>(src.data() + 8 * srcIdx));
        break;
      }
    }
    case FPrec::F64: {
      f64 *d = reinterpret_cast<f64 *>(dst.data() + 8 * dstIdx);
      switch (srcPrec) {
      case FPrec::F16:
        *d = static_cast<double>(
            *reinterpret_cast<const f16 *>(src.data() + 2 * srcIdx));
      case FPrec::F32:
        *d = f32(*reinterpret_cast<const f32 *>(src.data() + 4 * srcIdx));
      case FPrec::F64:
        std::unreachable();
        break;
      }
    }
    }
  }

  static std::size_t getIndexOf(WeightTensorLayout layout, std::size_t k,
                                std::size_t c, std::size_t s, std::size_t r,
                                std::size_t outputChannels,
                                std::size_t inputChannels,
                                std::size_t kernelWidth,
                                std::size_t kernelHeight) {
    switch (layout) {
    case WeightTensorLayout::KRSC:
      return k * (kernelWidth * kernelHeight * inputChannels) +
             r * (kernelWidth * inputChannels) + s * (inputChannels) + c;
    case vkcnn::WeightTensorLayout::KCRS:
      return k * (inputChannels * kernelHeight * kernelWidth) +
             c * (kernelHeight * kernelWidth) + r * kernelWidth + s;
    case vkcnn::WeightTensorLayout::RSCK:
      return r * (kernelWidth * inputChannels * outputChannels) +
             s * (inputChannels * outputChannels) + c * outputChannels + k;
    default:
      throw std::runtime_error(
          "Dynamic weight tensor does not support this layout.");
    }
  }

  WeightTensorLayout m_layout;
  FPrec m_precision;
  unsigned int m_outputChannels;
  unsigned int m_inputChannels;
  unsigned int m_kernelWidth;
  unsigned int m_kernelHeight;
  std::vector<std::byte> m_storage;
};

} // namespace vkcnn::host
