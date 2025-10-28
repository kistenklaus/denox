#include "Tensor.hpp"
#include "denox/common/types.hpp"
#include "dlpack/dlpack.h"
#include "f16.hpp"
#include <absl/strings/internal/str_format/extension.h>
#include <dlpack/dlpack.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <new>
#include <pycapsule.h>
#include <stdexcept>

namespace pydenox {

static denox::DataType parse_data_type_code(const DLDataType &type) {
  if (type.lanes != 1) {
    throw std::runtime_error(
        "Vectorized DLPack types (lanes != 1) are not supported.");
  }

  switch (type.code) {
  case kDLFloat:
    switch (type.bits) {
    case 16:
      return denox::DataType::Float16;
    case 32:
      return denox::DataType::Float32;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported floating-point bit width: {}", type.bits));
    }

  case kDLUInt:
    switch (type.bits) {
    case 8:
      return denox::DataType::Uint8;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported unsigned int bit width: {}", type.bits));
    }

  case kDLInt:
    switch (type.bits) {
    case 8:
      return denox::DataType::Int8;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported signed int bit width: {}", type.bits));
    }

  default:
    throw std::runtime_error(fmt::format(
        "Unsupported DLPack data type code: {}", static_cast<int>(type.code)));
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

    denox::DataType dldatatype = parse_data_type_code(t.dtype);
    if (dtype == denox::DataType::Auto) {
      dtype = dldatatype;
    }
    if (dtype != dldatatype) {
      throw std::runtime_error("Data type mismatch");
    }

    // TODO type and layout convertion.

    return Tensor{buffer, batch, H, W, C, dtype, layout, t.ndim};
  }
  throw std::runtime_error("Failed to parse tensor");
}

static std::size_t hwc_index(std::size_t N, std::size_t H, std::size_t W,
                             std::size_t C, std::size_t n, std::size_t h,
                             std::size_t w, std::size_t c) {
  return n * (H * W * C) + h * (W * C) + w * C + c;
}

static std::size_t chw_index(std::size_t N, std::size_t H, std::size_t W,
                             std::size_t C, std::size_t n, std::size_t h,
                             std::size_t w, std::size_t c) {
  return n * (H * W * C) + c * (H * W) + h * W + w;
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
    return *this;
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
    throw std::runtime_error("not implemented (chwc8)");
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
    throw std::runtime_error("not implemented (chwc8)");
  }
  assert(new_idx != nullptr);

  const std::byte *src_ptr = static_cast<const std::byte *>(m_data);
  std::byte *dst_ptr = static_cast<std::byte *>(buf);

  // NOTE obviously this has horrible performance.
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
  return Tensor{static_cast<void *>(dst_ptr), m_batchSize,
                m_height,
                m_width,
                m_channels,
                new_dtype,
                new_layout,
                m_rank};
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
    throw std::runtime_error("not implemented (dtype_size)");
  }
}
Tensor::Tensor(void *data, std::size_t N, std::size_t H, std::size_t W,
               std::size_t C, denox::DataType dtype, denox::Layout layout,
               int rank)
    : m_data(data), m_batchSize(N), m_height(H), m_width(W), m_channels(C),
      m_rank(rank), m_dtype(dtype), m_layout(layout) {
  if (m_rank != 3 && m_rank != 4) {
    throw std::runtime_error(
        "not supported. denox only supports tensors with rank 3 or 4.");
  }
}
Tensor::Tensor(const Tensor &o) {
  std::size_t size = o.byte_size();
  m_data = std::malloc(size);
  std::memcpy(m_data, o.m_data, size);
  m_batchSize = o.m_batchSize;
  m_height = o.m_height;
  m_width = o.m_width;
  m_channels = o.m_channels;
  m_dtype = o.m_dtype;
  m_layout = o.m_layout;
  m_rank = o.m_rank;
}
Tensor &Tensor::operator=(const Tensor &o) {
  if (this == &o) {
    return *this;
  }
  release();
  std::size_t size = o.byte_size();
  m_data = std::malloc(size);
  std::memcpy(m_data, o.m_data, size);
  m_batchSize = o.m_batchSize;
  m_height = o.m_height;
  m_width = o.m_width;
  m_channels = o.m_channels;
  m_dtype = o.m_dtype;
  m_layout = o.m_layout;
  m_rank = o.m_rank;
  return *this;
}
Tensor::Tensor(Tensor &&o)
    : m_data(std::exchange(o.m_data, nullptr)),
      m_batchSize(std::exchange(o.m_batchSize, 0)),
      m_height(std::exchange(o.m_height, 0)),
      m_width(std::exchange(o.m_width, 0)),
      m_channels(std::exchange(o.m_channels, 0)),
      m_rank(std::exchange(o.m_rank, 0)),
      m_dtype(std::exchange(o.m_dtype, denox::DataType::Auto)),
      m_layout(std::exchange(o.m_layout, denox::Layout::Undefined)) {}
Tensor &Tensor::operator=(Tensor &&o) {
  if (this == &o) {
    return *this;
  }
  release();
  std::swap(m_data, o.m_data);
  std::swap(m_batchSize, o.m_batchSize);
  std::swap(m_height, o.m_height);
  std::swap(m_width, o.m_width);
  std::swap(m_channels, o.m_channels);
  std::swap(m_dtype, o.m_dtype);
  std::swap(m_layout, o.m_layout);
  std::swap(m_rank, o.m_rank);
  return *this;
}
void Tensor::release() {
  if (m_data != nullptr) {
    std::free(m_data);
    m_data = nullptr;
    m_batchSize = 0;
    m_height = 0;
    m_width = 0;
    m_channels = 0;
    m_dtype = denox::DataType::Auto;
    m_layout = denox::Layout::Undefined;
  }
}
std::size_t Tensor::byte_size() const {
  return m_batchSize * m_height * m_width * m_channels * dtype_size(m_dtype);
}

Tensor Tensor::make(const void *data, std::size_t batchSize, std::size_t height,
                    std::size_t width, std::size_t channels,
                    denox::DataType dtype, denox::Layout layout, int rank) {
  std::size_t byteSize =
      batchSize * height * width * channels * dtype_size(dtype);
  void *ptr = std::malloc(byteSize);
  std::memcpy(ptr, data, byteSize);
  return Tensor{ptr, batchSize, height, width, channels, dtype, layout, rank};
}

void Tensor::define(pybind11::module_ &m) {
  pybind11::class_<pydenox::Tensor>(m, "Tensor")
      .def("__dlpack__", &Tensor::to_dlpack)
      .def("__dlpack_device__",
           [](const pydenox::Tensor &self) {
             // CPU tensors only
             return std::make_pair(static_cast<int>(kDLCPU), 0);
           })
      .def("size", &Tensor::size, pybind11::arg("dim"))
      .def("dim", &Tensor::rank)
      .def("batch", &Tensor::batchSize)
      .def("height", &Tensor::height)
      .def("width", &Tensor::width)
      .def("channels", &Tensor::channels)
      .def("dtype", &Tensor::dtype)
      .def("layout", &Tensor::layout);
}

static DLDataType to_dl_dtype(denox::DataType t) {
  switch (t) {
  case denox::DataType::Float16:
    return {kDLFloat, 16, 1};
  case denox::DataType::Float32:
    return {kDLFloat, 32, 1};
  case denox::DataType::Int8:
    return {kDLInt, 8, 1};
  case denox::DataType::Uint8:
    return {kDLUInt, 8, 1};
  default:
    return {kDLFloat, 32, 1};
  }
}

pybind11::capsule Tensor::to_dlpack() const {
  auto dl = new DLManagedTensor;
  std::memset(dl, 0, sizeof(DLManagedTensor));

  void *data = malloc(byte_size());
  std::memcpy(data, m_data, byte_size());

  DLDevice device{kDLCPU, 0};
  dl->dl_tensor.device = device;
  dl->dl_tensor.dtype = to_dl_dtype(m_dtype);
  dl->dl_tensor.ndim = m_rank;
  dl->dl_tensor.data = data;
  dl->dl_tensor.byte_offset = 0;

  assert(m_rank == 3 || m_rank == 4);
  auto *shape = new int64_t[m_rank];
  std::size_t o = m_rank == 3 ? 0 : 1;
  if (m_rank == 4) {
    shape[0] = static_cast<std::int64_t>(m_batchSize);
  }
  if (m_layout == denox::Layout::CHW) {
    shape[o + 0] = static_cast<int64_t>(m_channels);
    shape[o + 1] = static_cast<int64_t>(m_height);
    shape[o + 2] = static_cast<int64_t>(m_width);
  } else if (m_layout == denox::Layout::HWC) { // HWC
    shape[o + 0] = static_cast<int64_t>(m_height);
    shape[o + 1] = static_cast<int64_t>(m_width);
    shape[o + 2] = static_cast<int64_t>(m_channels);
  } else {
    throw std::runtime_error("layout not implemented");
  }
  dl->dl_tensor.shape = shape;
  dl->dl_tensor.strides = nullptr;

  // Single, canonical cleanup path:
  dl->deleter = [](DLManagedTensor *self) {
    if (!self)
      return;
    std::free(self->dl_tensor.data);
    delete[] self->dl_tensor.shape;
    delete self;
  };

  return pybind11::capsule(dl, "dltensor", [](PyObject *obj) {
    // Capsule may already be renamed to "used_dltensor" by PyTorch
    const char *name = PyCapsule_GetName(obj);
    if (!name)
      return;

    if (std::strcmp(name, "used_dltensor") == 0) {
      // Ownership already transferred — don't touch anything
      return;
    }

    if (std::strcmp(name, "dltensor") != 0) {
      // Unknown or corrupted capsule name — skip cleanup for safety
      return;
    }

    auto *self = reinterpret_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(obj, "dltensor"));
    if (self && self->deleter)
      self->deleter(self);
  });
}

std::size_t Tensor::size(int dim) const {
  switch (m_layout) {
  case denox::Layout::HWC:
    assert(m_rank == 3 || m_rank == 4);
    if (m_rank == 3) {
      switch (dim) {
      case 0:
        return m_height;
      case 1:
        return m_width;
      case 2:
        return m_channels;
      default:
        throw std::runtime_error("dim out of bound ");
      }
    } else {
      switch (dim) {
      case 0:
        return m_batchSize;
      case 1:
        return m_height;
      case 2:
        return m_width;
      case 3:
        return m_channels;
      default:
        throw std::runtime_error("dim out of bound");
      }
    }
    break;
  case denox::Layout::CHW:
    assert(m_rank == 3 || m_rank == 4);
    if (m_rank == 3) {
      switch (dim) {
      case 0:
        return m_channels;
      case 1:
        return m_height;
      case 2:
        return m_width;
      default:
        throw std::runtime_error("dim out of bound");
      }
    } else {
      switch (dim) {
      case 0:
        return m_batchSize;
      case 1:
        return m_channels;
      case 2:
        return m_height;
      case 3:
        return m_channels;
      default:
        throw std::runtime_error("dim out of bound");
      }
    }
    break;
  case denox::Layout::Undefined:
    throw std::runtime_error("unreachable");
  default:
  case denox::Layout::CHWC8:
    throw std::runtime_error("not implemented (CHWC8)");
  }
}
} // namespace pydenox
