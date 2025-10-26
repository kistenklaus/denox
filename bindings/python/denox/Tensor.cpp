#include "Tensor.hpp"
#include "denox/common/types.hpp"
#include "f16.hpp"
#include <absl/strings/internal/str_format/extension.h>
#include <dlpack/dlpack.h>
#include <memory>
#include <new>
#include <pycapsule.h>
#include <stdexcept>

namespace pydenox {

static denox::DataType parse_data_type_code(std::uint8_t code) {
  switch (code) {
  case DLDataTypeCode::kDLFloat:
    return denox::DataType::Float32;
  case DLDataTypeCode::kDLBfloat:
    return denox::DataType::Float16;
  default:
    throw std::runtime_error(
        "Failed to convert DLDataTypeCode to denox datatype.");
  }
}

Tensor Tensor::from(pybind11::object obj, denox::DataType dtype,
                    denox::Layout layout) {
  if (pybind11::hasattr(obj, "__dlpack__") ||
      pybind11::hasattr(obj, "to_dlpack")) {
    pybind11::object to_dlpack = pybind11::hasattr(obj, "to_dlpack")
                                     ? obj.attr("to_dlpack")
                                     : obj.attr("__dlpack__");
    pybind11::object capsule = to_dlpack();
    auto *unmanaged = reinterpret_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    auto deleter = [](DLManagedTensor *tensor) {};

    std::unique_ptr<DLManagedTensor, decltype(deleter)> managed(unmanaged,
                                                                deleter);
    if (!unmanaged || !unmanaged->dl_tensor.data) {
      throw std::runtime_error("Invalid DLPack tensor");
    }
    const DLTensor &t = unmanaged->dl_tensor;
    if (t.device.device_type != kDLCPU) {
      if (pybind11::hasattr(obj, "cpu")) {
        return Tensor::from(obj.attr("cpu"), dtype, layout);
      }
      throw std::runtime_error("GPU tensors not supported, call transfer to "
                               "cpu before calling Module(input).");
    }

    if (t.ndim < 3 || t.ndim > 4) {
      throw std::runtime_error("Tensor must be 3D or 4D");
    }
    // t.dtype.code

    std::size_t batch;
    std::size_t H;
    std::size_t W;
    std::size_t C;

    switch (layout) {
    case denox::Layout::Undefined:
      throw std::runtime_error("Undefined is not supported here.");
    case denox::Layout::HWC:
      batch = (t.ndim == 4) ? t.shape[0] : 1;
      H = t.shape[t.ndim - 3];
      W = t.shape[t.ndim - 2];
      C = t.shape[t.ndim - 1];
      break;
    case denox::Layout::CHW:
      batch = (t.ndim == 4) ? t.shape[0] : 1;
      C = t.shape[t.ndim - 3];
      H = t.shape[t.ndim - 2];
      W = t.shape[t.ndim - 1];
      break;
    case denox::Layout::CHWC8:
      throw std::runtime_error("CHWC8 layout is not supported here");
    }

    std::size_t elemSize = t.dtype.bits / 8;
    std::size_t totalSize = batch * H * W * C * elemSize;

    void *buffer = std::malloc(totalSize);
    if (!buffer) {
      throw std::bad_alloc();
    }
    std::memcpy(buffer, t.data, totalSize);

    denox::DataType dldatatype = parse_data_type_code(t.dtype.code);
    if (dtype == denox::DataType::Auto) {
      dtype = dldatatype;
    }
    if (dtype != dldatatype) {
      throw std::runtime_error("Data type mismatch");
    }

    // TODO type and layout convertion.

    return Tensor{
        buffer, batch, H, W, C, dtype, layout,
    };
  }
  throw std::runtime_error("Failed to parse tensor");
}

pybind11::object Tensor::to() const {
  throw std::runtime_error("not implemented");
}

static std::size_t hwc_index(std::size_t N, std::size_t H, std::size_t W,
                             std::size_t C, std::size_t n, std::size_t h,
                             std::size_t w, std::size_t c) {
  return n * (H * W * C) + h * (W * C) + w * C + c;
}

static std::size_t chw_index(std::size_t N, std::size_t H, std::size_t W,
                             std::size_t C, std::size_t n, std::size_t h,
                             std::size_t w, std::size_t c) {
  return n * (H * W * C) + c * (W * C) + h * W + w;
}

typedef std::size_t (*layout_index)(std::size_t N, std::size_t H, std::size_t W,
                                    std::size_t C, std::size_t n, std::size_t h,
                                    std::size_t w, std::size_t c);

Tensor Tensor::transform(denox::DataType new_dtype,
                         denox::Layout new_layout) const {
  if (new_dtype == denox::DataType::Auto) {
    throw std::invalid_argument(
        "Tensor::transform requires concrete new_dtype argument, got Auto.");
  }
  if (new_layout == denox::Layout::Undefined) {
    throw std::invalid_argument("Tensor::transform requires concrete "
                                "new_layout argument, got Undefined.");
  }
  if (new_dtype == m_dtype && new_layout == m_layout) {
    return *this; // TODO requires copy constructor.
  }

  std::size_t new_dtype_size = Tensor::dtype_size(new_dtype);
  std::size_t old_dtype_size = Tensor::dtype_size(m_dtype);

  std::size_t totalSize =
      new_dtype_size * m_width * m_height * m_channels * m_batchSize;
  void *buf = std::malloc(totalSize);

  layout_index old_idx = nullptr;
  switch (m_layout) {
  case denox::Layout::Undefined:
    throw std::runtime_error("unreachable");
  case denox::Layout::HWC:
    old_idx = hwc_index;
    break;
  case denox::Layout::CHW:
    old_idx = chw_index;
    break;
  case denox::Layout::CHWC8:
    throw std::runtime_error("not implemented");
  }
  assert(old_idx != nullptr);
  layout_index new_idx = nullptr;
  switch (new_layout) {
  case denox::Layout::Undefined:
    throw std::runtime_error("unreachable");
  case denox::Layout::HWC:
    new_idx = hwc_index;
    break;
  case denox::Layout::CHW:
    new_idx = chw_index;
    break;
  case denox::Layout::CHWC8:
    throw std::runtime_error("not implemented");
  }

  const std::byte *src_ptr = static_cast<const std::byte *>(m_data);
  std::byte *dst_ptr = static_cast<std::byte *>(buf);

  // NOTE obviously this is horrible performance =^).
  for (std::size_t n = 0; n < m_batchSize; ++n) {
    for (std::size_t h = 0; h < m_height; ++h) {
      for (std::size_t w = 0; w < m_width; ++w) {
        for (std::size_t c = 0; c < m_channels; ++c) {
          std::size_t oldOffset =
              old_idx(m_batchSize, m_height, m_width, m_channels, n, h, w, c);
          std::size_t newOffset =
              new_idx(m_batchSize, m_height, m_width, m_channels, n, h, w, c);
          const void *src = src_ptr + oldOffset * old_dtype_size;
          void *dst = dst_ptr + newOffset * new_dtype_size;
          switch (new_dtype) {
          case denox::Float16: {
            f16 v;
            switch (m_dtype) {
            case denox::Float16:
              v = *reinterpret_cast<const f16 *>(src);
              break;
            case denox::Float32:
              v = f16(*reinterpret_cast<const float *>(src));
              break;
            case denox::Uint8:
            case denox::Int8:
            case denox::Auto:
            default:
              throw std::runtime_error("unreachable");
            }
            *reinterpret_cast<f16 *>(dst) = v;
            break;
          }
          case denox::Float32: {
            float v;
            switch (m_dtype) {
            case denox::Float16:
              v = static_cast<float>(*reinterpret_cast<const f16 *>(src));
              break;
            case denox::Float32:
              v = *reinterpret_cast<const float *>(src);
              break;
            case denox::Auto:
            case denox::Uint8:
            case denox::Int8:
            default:
              throw std::runtime_error("unreachable");
              break;
            }
            *reinterpret_cast<float *>(dst) = v;
            break;
          }
          case denox::Auto:
          case denox::Uint8:
          case denox::Int8:
            throw std::runtime_error("unreachable");
          }
        }
      }
    }
  }

  throw std::runtime_error("not implemented");
}

std::size_t Tensor::dtype_size(denox::DataType dtype) {
  switch (dtype) {
  case denox::Auto:
    throw std::runtime_error("invalid state");
  case denox::Float16:
    return 2;
  case denox::Float32:
    return 4;
  case denox::Uint8:
    return 1;
  case denox::Int8:
    return 1;
  default:
    throw std::runtime_error("not implemented");
  }
}
Tensor::Tensor(void *data, std::size_t N, std::size_t H, std::size_t W,
               std::size_t C, denox::DataType dtype, denox::Layout layout)
    : m_data(data), m_batchSize(N), m_height(H), m_width(W), m_channels(C),
      m_dtype(dtype), m_layout(layout) {}
} // namespace pydenox
