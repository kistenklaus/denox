#pragma once

#include "vkcnn/host/ImageTensorLayout.hpp"
#include "vkcnn/host/floatx.hpp"
#include "vkcnn/host/fprec.hpp"
#include <cstring>
#include <functional>
#include <glm/vec3.hpp>
#include <print>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace vkcnn::host {

class DynamicImageTensor {
public:
  DynamicImageTensor(ImageTensorLayout layout, FPrec precision, std::size_t w,
                     std::size_t h, std::size_t c)
      : m_layout(layout), m_precision(precision), m_width(w), m_height(h),
        m_channels(c),
        m_storage(m_width * m_height * m_channels * FPrec_Size(precision)) {}

  template <typename T>
  void set(std::size_t w, std::size_t h, std::size_t c, T v) {
    return DynamicImageTensor::set<T>(
        m_storage, getIndexOf(m_layout, w, h, c, m_width, m_height, m_channels),
        m_precision, v);
  }

  template <typename T> T get(std::size_t w, std::size_t h, std::size_t c) {
    return DynamicImageTensor::get<T>(
        m_storage, getIndexOf(m_layout, w, h, c, m_width, m_height, m_channels),
        m_precision);
  }

  void transformLayout(ImageTensorLayout newLayout) {
    if (newLayout == m_layout) {
      return;
    }
    auto oldStorage = m_storage;
    auto oldLayout = m_layout;
    for (std::size_t w = 0; w < m_width; ++w) {
      for (std::size_t h = 0; h < m_height; ++h) {
        for (std::size_t c = 0; c < m_channels; ++c) {
          copy(m_storage,
               getIndexOf(newLayout, w, h, c, m_width, m_height, m_channels),
               oldStorage,
               getIndexOf(oldLayout, w, h, c, m_width, m_height, m_channels),
               m_precision);
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
    std::size_t size = m_width * m_height * m_channels;
    m_storage.resize(size * fSize);
    for (std::size_t i = 0; i < size; ++i) {
      cast(m_storage, i, newPrecision, oldStorage, i, oldPrec);
    }
  }

  template <typename T> void fill(std::function<T(std::size_t)> gen) {
    std::size_t size = m_width * m_height * m_channels;
    for (std::size_t i = 0; i < size; ++i) {
      DynamicImageTensor::set<T>(m_storage, i, m_precision, gen(i));
    }
  }

  inline unsigned int w() const { return m_width; }
  inline unsigned int h() const { return m_height; }
  inline unsigned int c() const { return m_channels; }
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

  static std::size_t getIndexOf(ImageTensorLayout layout, std::size_t w,
                                std::size_t h, std::size_t c, std::size_t width,
                                std::size_t height, std::size_t channels) {
    switch (layout) {
    case ImageTensorLayout::HWC:
      return h * (width * channels) + w * (channels) + c;
      break;
    case ImageTensorLayout::CHW:
      return c * (height * width) + h * (width) + w;
      break;
    default:
      throw std::runtime_error(
          "DynamicTensor does not yet support this layout.");
    }
  }

  ImageTensorLayout m_layout;
  FPrec m_precision;
  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_channels;
  std::vector<std::byte> m_storage;
};

} // namespace vkcnn::host
