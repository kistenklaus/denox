#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include "vkcnn/tensor/ImageTensorLayout.hpp"
#include "vkcnn/tensor/float_types.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cassert>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <functional>
#include <glm/vec3.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace vkcnn {

class ImageTensor {
private:
  using Alloc = std::allocator<std::byte>;
  using ATraits = std::allocator_traits<Alloc>;
  struct Shape {
    ImageTensorLayout layout;
    FloatType precision;
    unsigned int width;
    unsigned int height;
    unsigned int channels;
  };
  struct Host {
    std::variant<std::nullptr_t, merian::BufferHandle, std::byte *> buffer;
    unsigned int size;
    [[no_unique_address]] Alloc allocator;
  };
  struct Device {
    merian::BufferHandle buffer;
    merian::ResourceAllocatorHandle allocator;
    bool dirty;
  };

public:
  ImageTensor(ImageTensorLayout layout, FloatType precision, std::size_t w,
              std::size_t h, std::size_t c)
      : m_shape(layout, precision, w, h, c),
        m_host(nullptr, w * h * c * Float_Size(precision), {}),
        m_device(nullptr, nullptr, true) {
    if (layout == ImageTensorLayout::CHWC8) {
      assert(c % 8 == 0);
    }
  }

  ~ImageTensor() {
    releaseHostStorage();
    releaseDeviceStorage();
  }

  template <typename T>
  void set(std::size_t w, std::size_t h, std::size_t c, T v) {
    std::byte *mapped = mapHostStorage();
    set<T>(mapped, getIndexOf(m_shape, w, h, c), m_shape.precision, v);
    unmapHostStorage();
    m_device.dirty = true;
  }

  template <typename T>
  T get(std::size_t w, std::size_t h, std::size_t c) const {
    if (std::holds_alternative<std::nullptr_t>(m_host.buffer)) {
      return T{};
    } else {
      std::byte *buffer = mapHostStorage();
      // std::println("Getting {}", getIndexOf(m_shape, w, h, c));
      T v = get<T>(buffer, getIndexOf(m_shape, w, h, c), m_shape.precision);
      unmapHostStorage();
      return v;
    }
  }

  /// Transforms the layout of the tensor.
  void transformLayout(ImageTensorLayout layout) {
    if (layout == m_shape.layout) {
      return;
    }
    if (layout == ImageTensorLayout::CHWC8) {
      assert(c() % 8 == 0);
    }
    bool hostAvailable = !std::holds_alternative<std::nullptr_t>(m_host.buffer);
    bool deviceAvailable = m_device.buffer != nullptr;

    if (!hostAvailable && !deviceAvailable) {
      // no need to move any data around because there is none.
      m_shape.layout = layout;
      return;
    }
    if (deviceAvailable && !hostAvailable) {
      throw std::logic_error("ImageTensor::transformLayout called from an "
                             "invalid state: Tensor is not host visible. We do "
                             "not implement device transformation.");
    }
    assert(hostAvailable);

    std::byte *oldStorage = mapHostStorage();

    // TODO: Smarter reallocation, here we might be able to allocate a new
    // vulkan buffer as well, if the previous buffer was a vulkan buffer.
    std::byte *newStorage = ATraits::allocate(m_host.allocator, m_host.size);
    Shape oldShape = m_shape;
    m_shape.layout = layout;
    for (std::size_t w = 0; w < m_shape.width; ++w) {
      for (std::size_t h = 0; h < m_shape.height; ++h) {
        for (std::size_t c = 0; c < m_shape.channels; ++c) {
          copy(newStorage, getIndexOf(m_shape, w, h, c), oldStorage,
               getIndexOf(oldShape, w, h, c), m_shape.precision);
        }
      }
    }
    unmapHostStorage();
    releaseHostStorage();
    m_device.dirty = true;
    m_host.buffer = newStorage;
  }

  void transformFloatType(FloatType precision) {
    if (precision == m_shape.precision) {
      return;
    }
    bool hostAvailable = !std::holds_alternative<std::nullptr_t>(m_host.buffer);
    bool deviceAvailable = m_device.buffer != nullptr;

    if (!hostAvailable && !deviceAvailable) {
      // no need to move any data around because there is none.
      m_shape.precision = precision;
      return;
    }
    if (deviceAvailable && !hostAvailable) {
      throw std::logic_error("ImageTensor::transformLayout called from an "
                             "invalid state: Tensor is not host visible. We do "
                             "not implement device transformation.");
    }
    assert(hostAvailable);

    std::byte *oldStorage = mapHostStorage();
    auto oldPrec = m_shape.precision;
    auto fSize = Float_Size(precision);
    std::size_t size = m_shape.width * m_shape.height * m_shape.channels;

    // TODO: Smarter reallocation, here we might be able to allocate a new
    // vulkan buffer as well.
    std::byte *newStorage = ATraits::allocate(m_host.allocator, size * fSize);

    for (std::size_t i = 0; i < size; ++i) {
      cast(newStorage, i, precision, oldStorage, i, oldPrec);
    }
    unmapHostStorage();
    releaseHostStorage();
    m_host.size = size * fSize;
    m_shape.precision = precision;
    m_host.buffer = newStorage;
    m_device.dirty = true;
  }

  template <typename T> void fill(std::function<T(std::size_t)> gen) {
    std::byte *mapped = mapHostStorage();
    std::size_t size = m_shape.width * m_shape.height * m_shape.channels;
    for (std::size_t i = 0; i < size; ++i) {
      set<T>(mapped, i, m_shape.precision, gen(i));
    }
    unmapHostStorage();
  }

  friend ImageTensor operator+(const ImageTensor &a, const ImageTensor &b) {
    assert(a.w() == b.w() && a.h() == b.h() && a.c() == b.c());
    ImageTensor out{ImageTensorLayout::HWC, FloatType::F32, a.w(), a.h(), a.c()};
    for (unsigned int y = 0; y < a.h(); ++y) {
      for (unsigned int x = 0; x < a.w(); ++x) {
        for (unsigned int c = 0; c < a.c(); ++c) {
          out.set<float>(x, y, c,
                         a.get<float>(x, y, c) + b.get<float>(x, y, c));
        }
      }
    }
    return out;
  }

  friend ImageTensor operator-(const ImageTensor &a, const ImageTensor &b) {
    assert(a.w() == b.w() && a.h() == b.h() && a.c() == b.c());
    ImageTensor out{ImageTensorLayout::HWC, FloatType::F32, a.w(), a.h(), a.c()};
    for (unsigned int y = 0; y < a.h(); ++y) {
      for (unsigned int x = 0; x < a.w(); ++x) {
        for (unsigned int c = 0; c < a.c(); ++c) {
          out.set<float>(x, y, c,
                         a.get<float>(x, y, c) - b.get<float>(x, y, c));
        }
      }
    }
    return out;
  }

  inline unsigned int w() const { return m_shape.width; }
  inline unsigned int h() const { return m_shape.height; }
  inline unsigned int c() const { return m_shape.channels; }
  inline FloatType precision() const { return m_shape.precision; }

  void enableDevice(const merian::ResourceAllocatorHandle &alloc) {
    m_device.allocator = alloc;
    requireStage(alloc);
    requireDevice(alloc);
  }

  void disableDevice() {
    bool hostAvailable = !std::holds_alternative<std::nullptr_t>(m_host.buffer);
    bool staged = std::holds_alternative<merian::BufferHandle>(m_host.buffer);
    bool deviceAvailable = m_device.buffer != nullptr;
    if (!hostAvailable && !deviceAvailable) {
      return;
    }

    if (!hostAvailable && deviceAvailable) {
      throw std::logic_error("Invalid state");
    }
    if (!staged) {
      throw std::logic_error("Invalid state: requires staging buffer");
    }
    assert(hostAvailable);
    assert(deviceAvailable);
    assert(staged);

    std::byte *newStorage = ATraits::allocate(m_host.allocator, m_host.size);
    std::byte *mapped = mapHostStorage();
    std::memcpy(newStorage, mapped, m_host.size);
    unmapHostStorage();
    releaseHostStorage();
    m_host.buffer = newStorage;
    releaseDeviceStorage();
  }

  void flushToDeviceAndWait(const merian::QueueHandle &queue,
                            const merian::CommandPoolHandle &cmdPool) {
    merian::CommandBufferHandle cmd =
        std::make_shared<merian::CommandBuffer>(cmdPool);
    cmd->begin();
    flushToDevice(cmd);
    cmd->end();
    queue->submit_wait(cmd);
  }

  void flushToDevice(const merian::CommandBufferHandle &cmd) {
    assert(m_device.buffer != nullptr);
    assert(std::holds_alternative<merian::BufferHandle>(m_host.buffer));
    if (m_device.dirty) {

      const auto &stage = std::get<merian::BufferHandle>(m_host.buffer);
      vk::BufferCopy copy{0, 0, m_host.size};

      cmd->barrier(vk::PipelineStageFlagBits::eHost,
                   vk::PipelineStageFlagBits::eTransfer,
                   stage->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                         vk::AccessFlagBits::eTransferRead));

      cmd->copy(stage, m_device.buffer, copy);
      cmd->barrier(
          vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eComputeShader,
          m_device.buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                          vk::AccessFlagBits::eShaderRead));
      m_device.dirty = false;
    }
  }

  void flushToHost(const merian::CommandBufferHandle &cmd) {
    assert(m_device.buffer != nullptr);
    assert(std::holds_alternative<merian::BufferHandle>(m_host.buffer));

    auto &stage = std::get<merian::BufferHandle>(m_host.buffer);
    vk::BufferCopy copy{0, 0, m_host.size};

    cmd->barrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        m_device.buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                        vk::AccessFlagBits::eTransferRead));

    cmd->copy(m_device.buffer, stage, copy);

    cmd->barrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
        m_device.buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                        vk::AccessFlagBits::eHostRead));
  }

  merian::BufferHandle getDeviceHandle() const { return m_device.buffer; }

  std::size_t byteSize() const { return m_host.size; }

  glm::uvec2 imageExtent() const {
    return glm::uvec2(m_shape.width, m_shape.height);
  }

  void releaseHost() { releaseHostStorage(); }

  void releaseDevice() { releaseDeviceStorage(); }

private:
  template <class... Ts> struct overloads : Ts... {
    using Ts::operator()...;
  };

  /// Implicity creates host storage on the fly.
  std::byte *mapHostStorage() const {
    const auto visitor =
        overloads{[&](const std::nullptr_t &) -> std::byte * {
                    std::byte *buffer =
                        ATraits::allocate(m_host.allocator, m_host.size);
                    std::memset(buffer, 0, m_host.size);
                    m_host.buffer = buffer;
                    return buffer;
                  },
                  [&](const merian::BufferHandle &b) -> std::byte * {
                    return b->get_memory()->map_as<std::byte>();
                  },
                  [&](std::byte *b) -> std::byte * { return b; }};
    return std::visit(visitor, m_host.buffer);
  }

  void unmapHostStorage() const {
    if (std::holds_alternative<merian::BufferHandle>(m_host.buffer)) {
      std::get<merian::BufferHandle>(m_host.buffer)->get_memory()->unmap();
    }
  }

  void releaseHostStorage() {
    if (std::holds_alternative<std::byte *>(m_host.buffer)) {
      std::byte *b = std::get<std::byte *>(m_host.buffer);
      ATraits::deallocate(m_host.allocator, b, m_host.size);
    }
    m_host.buffer = nullptr;
  }

  /// Ensures that a the host memory is stored in a staging buffer
  /// and not a simply cpu allocation.
  void requireStage(const merian::ResourceAllocatorHandle &alloc) {
    const auto visitor = overloads{
        [&](const std::nullptr_t &) {
          merian::BufferHandle buffer = createStagingBuffer(alloc);
          // unlucky situation.
          std::byte *mapped = buffer->get_memory()->map_as<std::byte>();
          std::memset(mapped, 0, m_host.size);
          buffer->get_memory()->unmap();
          m_host.buffer = buffer;
        },
        [&](const merian::BufferHandle &) {}, //
        [&](std::byte *b) {
          merian::BufferHandle buffer = createStagingBuffer(alloc);
          // unlucky situation.
          std::byte *mapped = buffer->get_memory()->map_as<std::byte>();
          std::memcpy(mapped, b, m_host.size);
          buffer->get_memory()->unmap();
          ATraits::deallocate(m_host.allocator, b, m_host.size);
          m_host.buffer = buffer;
        }};
    std::visit(visitor, m_host.buffer);
  }

  /// Ensures that a the host memory is stored in a staging buffer
  /// and not a simply cpu allocation.
  void requireDevice(const merian::ResourceAllocatorHandle &alloc) {
    if (m_device.buffer == nullptr) {
      m_device.buffer = alloc->createBuffer(
          m_host.size,
          vk::BufferUsageFlagBits::eTransferDst |
              vk::BufferUsageFlagBits::eTransferSrc |
              vk::BufferUsageFlagBits::eStorageBuffer,
          merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);
      assert(m_device.buffer);
    }
  }
  void releaseDeviceStorage() { m_device.buffer = nullptr; }

  merian::BufferHandle
  createStagingBuffer(const merian::ResourceAllocatorHandle &alloc) {
    return alloc->createBuffer(
        m_host.size,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);
  }

  template <typename T>
  __attribute__((always_inline)) static inline T
  get(const std::byte *storage, std::size_t idx, FloatType precision) {
    switch (precision) {
      case FloatType::F16:
      if constexpr (std::same_as<T, f16>) {
        return *reinterpret_cast<const f16 *>(storage + idx * 2);
      } else {
        return static_cast<T>(
            *reinterpret_cast<const f16 *>(storage + idx * 2));
      }
    case FloatType::F32:
      return T(*reinterpret_cast<const f32 *>(storage + idx * 4));
    case FloatType::F64:
      return T(*reinterpret_cast<const f64 *>(storage + idx * 8));
    }
    throw std::invalid_argument("Invalid precision");
  }

  template <typename T>
  __attribute__((always_inline)) static inline void
  set(std::byte *storage, std::size_t idx, FloatType precision, T v) {
    switch (precision) {
    case FloatType::F16:
      *reinterpret_cast<f16 *>(storage + idx * 2) = f16(v);
      break;
    case FloatType::F32:
      if constexpr (std::same_as<T, f16>) {
        *reinterpret_cast<f32 *>(storage + idx * 4) = static_cast<f32>(v);
      } else {
        *reinterpret_cast<f32 *>(storage + idx * 4) = f32(v);
      }
      break;
    case FloatType::F64:
      if constexpr (std::same_as<T, f16>) {
        *reinterpret_cast<f64 *>(storage + idx * 8) = static_cast<f64>(v);
      } else {
        *reinterpret_cast<f64 *>(storage + idx * 8) = f64(v);
      }
      break;
    }
  }

  __attribute__((always_inline)) static inline void
  copy(std::byte *dst, std::size_t dstIdx, const std::byte *src,
       std::size_t srcIdx, FloatType precision) {
    std::size_t fsize = Float_Size(precision);
    // fmt::println("Copying {} -> {}", dstIdx * fsize, srcIdx * fsize);
    std::memcpy(dst + dstIdx * fsize, src + srcIdx * fsize, fsize);
  }

  /// Hopefully this is inlined and the branching is put outside of the loop.
  __attribute__((always_inline)) static inline void
  cast(std::byte *dst, std::size_t dstIdx, FloatType dstPrec, const std::byte *src,
       std::size_t srcIdx, FloatType srcPrec) {
    if (dstPrec == srcPrec) {
      return copy(dst, dstIdx, src, srcIdx, srcPrec);
    }
    switch (dstPrec) {
    case FloatType::F16: {
      f16 *d = reinterpret_cast<f16 *>(dst + 2 * dstIdx);
      switch (srcPrec) {
      case FloatType::F16:
        std::unreachable();
        break;
      case FloatType::F32:
        *d = f16(*reinterpret_cast<const f32 *>((src + 4 * srcIdx)));
        break;
      case FloatType::F64:
        *d = f16(*reinterpret_cast<const f64 *>((src + 8 * srcIdx)));
        break;
      }
      break;
    }
    case FloatType::F32: {
      f32 *d = reinterpret_cast<f32 *>(dst + 4 * dstIdx);
      switch (srcPrec) {
      case FloatType::F16:
        *d = static_cast<f32>(*reinterpret_cast<const f16 *>(src + 2 * srcIdx));
        break;
      case FloatType::F32:
        std::unreachable();
        break;
      case FloatType::F64:
        *d = f64(*reinterpret_cast<const f64 *>(src + 8 * srcIdx));
        break;
      }
      break;
    }
    case FloatType::F64: {
      f64 *d = reinterpret_cast<f64 *>(dst + 8 * dstIdx);
      switch (srcPrec) {
      case FloatType::F16:
        *d = static_cast<double>(
            *reinterpret_cast<const f16 *>(src + 2 * srcIdx));
        break;
      case FloatType::F32:
        *d = f32(*reinterpret_cast<const f32 *>(src + 4 * srcIdx));
        break;
      case FloatType::F64:
        std::unreachable();
        break;
      }
      break;
    }
    }
  }

  static std::size_t getIndexOf(const Shape &shape, std::size_t w,
                                std::size_t h, std::size_t c) {
    switch (shape.layout) {
    case ImageTensorLayout::HWC:
      return h * (shape.width * shape.channels) + w * (shape.channels) + c;
      break;
    case ImageTensorLayout::CHW:
      return c * (shape.height * shape.width) + h * (shape.width) + w;
      break;
    case ImageTensorLayout::CHWC8:
      return (c >> 3) * (shape.height * shape.width * 8) +
             h * (shape.width * 8) + w * 8 + (c & 0x7);
      break;
    default:
      throw std::runtime_error(
          "DynamicTensor does not yet support this layout.");
    }
  }
  Shape m_shape;
  mutable Host m_host;
  Device m_device;
};

} // namespace vkcnn

template <> struct fmt::formatter<vkcnn::ImageTensor> {
  constexpr auto parse(format_parse_context &ctx) const { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const vkcnn::ImageTensor &tensor, FormatContext &ctx) const {
    std::ostringstream oss;

    const auto w = tensor.w();
    const auto h = tensor.h();
    const auto c = tensor.c();
    auto printValue = [&](std::ostream &os, std::size_t wi, std::size_t hi,
                          std::size_t ci) {
      float value = tensor.get<float>(wi, hi, ci);
      os << std::fixed << std::setw(6) << std::setprecision(2) << value;
    };

    for (std::size_t ci = 0; ci < c; ++ci) {
      oss << "Channel: " << ci << "\n";
      for (std::size_t hi = 0; hi < h; ++hi) {
        for (std::size_t wi = 0; wi < w; ++wi) {
          printValue(oss, wi, hi, ci);
          oss << ' ';
        }
        oss << '\n';
      }
    }

    return fmt::format_to(ctx.out(), "{}", oss.str());
  }
};
