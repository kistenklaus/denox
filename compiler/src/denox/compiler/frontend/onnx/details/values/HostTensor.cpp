#include "denox/compiler/frontend/onnx/details/values/HostTensor.hpp"
#include "denox/io/fs/File.hpp"
#include <stdexcept>

#include <onnx.pb.h>

namespace denox::onnx::details {

HostTensor::HostTensor(TensorShape shape,
                       memory::shared_ptr<HostTensorStorage> storage)
    : m_shape(shape), m_view(TensorViewDesc::Identity(shape)),
      m_store(std::move(storage)) {}
const memory::shared_ptr<HostTensorStorage> &HostTensor::storage() const {
  return m_store;
}

bool HostTensor::isContiguous() const {
  if (!m_shape.isConstant())
    return false;
  return m_view.isRowMajorContiguous(m_shape.toU64());
}

std::size_t HostTensor::sizeBytesIfStatic() const {
  if (!m_shape.isConstant())
    return 0;
  auto elems = m_shape.toU64();
  std::size_t n = 1;
  for (auto e : elems)
    n *= static_cast<std::size_t>(e);
  return n * elemSize();
}

std::size_t HostTensor::byteOffset() const {
  assert(m_view.isConstant());
  return static_cast<std::size_t>(m_view.offset().constant()) * elemSize();
}

const void *HostTensor::data() const {
  return static_cast<const std::byte *>(m_store->data()) + byteOffset();
}

void *HostTensor::data() {
  return static_cast<std::byte *>(m_store->data()) + byteOffset();
}

HostTensor HostTensor::withView(TensorShape newShape,
                                TensorViewDesc newView) const {
  return HostTensor(std::move(newShape), std::move(newView), m_store);
}

HostTensor HostTensor::permute(memory::span<const int64_t> perm) const {
  auto newShape = m_shape.permute(perm);
  auto newView = m_view.permute(perm);
  return withView(std::move(newShape), std::move(newView));
}

HostTensor HostTensor::unsqueeze(std::size_t axis) const {
  auto newShape = m_shape.unsqueeze(axis);
  auto newView = m_view.unsqueeze(axis);
  return withView(std::move(newShape), std::move(newView));
}

HostTensor HostTensor::squeeze(std::size_t axis) const {
  auto newShape = m_shape.squeeze(axis);
  auto newView = m_view.squeeze(axis);
  return withView(std::move(newShape), std::move(newView));
}

HostTensor HostTensor::materializeContiguous() const {
  if (!m_shape.isConstant()) {
    throw std::runtime_error("materializeContiguous: shape must be static");
  }

  const std::size_t elem = elemSize();
  const auto sizes = m_shape.toU64();
  const std::size_t rank = sizes.size();

  std::size_t totalElems = 1;
  for (auto s : sizes)
    totalElems *= static_cast<std::size_t>(s);
  const std::size_t bytes = totalElems * elem;

  if (bytes == 0) {
    memory::shared_ptr<HostTensorStorage> empty;
    if (type() == Dtype::String) {
      empty = std::make_shared<HostTensorStorage>(
          HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0));
    } else {
      empty = std::make_shared<HostTensorStorage>(
          HostTensorStorage::Raw(type(), nullptr, 0));
    }
    return HostTensor(m_shape, std::move(empty));
  }

  if (type() != Dtype::String && isContiguous() &&
      m_view.offset().isConstant() && m_view.offset().constant() == 0) {
    return *this;
  }

  const auto vstr = m_view.strides();
  memory::vector<long long> strideBytes(vstr.size());
  for (size_t i = 0; i < vstr.size(); ++i) {
    assert(vstr[i].isConstant() &&
           "materializeContiguous: view must be constant for now");
    const long long si = static_cast<long long>(vstr[i].constant());
    strideBytes[i] = si * static_cast<long long>(elem);
  }
  assert(m_view.offset().isConstant());
  const long long base = static_cast<long long>(m_view.offset().constant()) *
                         static_cast<long long>(elem);

  const auto *srcBase = static_cast<const std::byte *>(m_store->data());

  auto advance_index = [&](memory::vector<std::size_t> &idx) -> bool {
    if (idx.empty())
      return false;
    size_t ax = idx.size();
    while (ax > 0) {
      --ax;
      if (++idx[ax] < sizes[ax])
        return true;
      idx[ax] = 0;
    }
    return false;
  };

  auto compute_offset_bytes =
      [&](memory::span<const std::size_t> idx) -> long long {
    long long off = base;
    for (size_t ax = 0; ax < idx.size(); ++ax) {
      off += strideBytes[ax] * static_cast<long long>(idx[ax]);
    }
    return off;
  };

  // 5) STRING dtype: deep-copy each element (no memcpy of pointer tables)
  if (type() == Dtype::String) {
    // Allocate output table of char* (totalElems entries)
    char **outTable =
        static_cast<char **>(std::malloc(totalElems * sizeof(char *)));
    if (!outTable)
      throw std::bad_alloc();

    std::size_t k = 0;

    auto copy_one = [&](long long offBytes) {
      assert(static_cast<std::size_t>(offBytes) % sizeof(const char *) == 0);

      const char *srcp{};
      std::memcpy(&srcp, srcBase + offBytes, sizeof srcp);
      if (!srcp) {
        outTable[k++] = nullptr;
        return;
      }

      const std::size_t len = std::strlen(srcp);
      if (len >= (std::numeric_limits<std::size_t>::max)() - 1)
        throw std::overflow_error("string too large");

      char *cp = static_cast<char *>(std::malloc(len + 1));
      if (!cp)
        throw std::bad_alloc();

      std::memcpy(cp, srcp, len + 1);
      outTable[k++] = cp;
    };

    if (rank == 0) {
      copy_one(base);
    } else {
      memory::vector<std::size_t> idx(rank, 0);
      while (true) {
        const long long off = compute_offset_bytes(idx);
        copy_one(off);
        if (!advance_index(idx))
          break;
      }
    }

    auto newStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::String, outTable, totalElems * sizeof(char *)));
    return HostTensor(m_shape, std::move(newStore));
  }

  void *dst = std::malloc(bytes);
  if (!dst)
    throw std::bad_alloc();

  auto copy_elementwise = [&]() {
    if (rank == 0) {
      std::memcpy(dst, srcBase + base, elem);
      return;
    }
    memory::vector<std::size_t> idx(rank, 0);
    std::byte *out = static_cast<std::byte *>(dst);
    while (true) {
      const long long off = compute_offset_bytes(idx);
      std::memcpy(out, srcBase + off, elem);
      out += elem;
      if (!advance_index(idx))
        break;
    }
  };

  auto copy_rows = [&]() {
    const bool innermostContig =
        (rank == 0) || strideBytes.empty() ||
        (strideBytes.back() == static_cast<long long>(elem));

    if (!innermostContig) {
      copy_elementwise();
      return;
    }

    const std::size_t rowElems = (rank == 0 ? 1 : sizes.back());
    const std::size_t rowBytes = rowElems * elem;

    if (rank <= 1) {
      std::memcpy(dst, srcBase + base, rowBytes);
      return;
    }

    memory::vector<std::size_t> idx(rank - 1, 0);
    std::byte *out = static_cast<std::byte *>(dst);

    while (true) {
      long long off = base;
      for (size_t ax = 0; ax < rank - 1; ++ax) {
        off += strideBytes[ax] * static_cast<long long>(idx[ax]);
      }
      const std::byte *src = srcBase + off;
      std::memcpy(out, src, rowBytes);
      out += rowBytes;

      if (idx.empty())
        break;
      size_t ax = idx.size();
      while (ax > 0) {
        --ax;
        if (++idx[ax] < sizes[ax])
          goto cont;
        idx[ax] = 0;
      }
      break;
    cont:;
    }
  };

  copy_rows();

  auto newStore = std::make_shared<HostTensorStorage>(
      HostTensorStorage::TakeOwnership(type(), dst, bytes));
  return HostTensor(m_shape, std::move(newStore));
}

std::size_t HostTensor::sizeElemsIfStatic() const {
  if (!m_shape.isConstant())
    return 0;
  std::size_t n = 1;
  for (auto d : m_shape.toU64())
    n *= static_cast<std::size_t>(d);
  return n;
}

HostTensor HostTensor::contiguous() const {
  return (isContiguous() && m_view.offset().isConstant() &&
          m_view.offset().constant() == 0)
             ? *this
             : materializeContiguous();
}

HostTensor HostTensor::reshape(const TensorShape &newShape) const {
  assert(isConstant() && "reshape needs static size");
  auto oldN = sizeElemsIfStatic();
  std::size_t newN = 1;
  for (auto d : newShape.toU64())
    newN *= static_cast<std::size_t>(d);
  if (oldN != newN)
    throw std::runtime_error("reshape: numel mismatch");

  if (!isContiguous() ||
      !(m_view.offset().isConstant() && m_view.offset().constant() == 0)) {
    return materializeContiguous().reshape(newShape);
  }
  auto newView = TensorViewDesc::Identity(newShape);
  return withView(newShape, std::move(newView));
}

HostTensor HostTensor::select(std::size_t axis, std::uint64_t index) const {
  auto g = m_shape.graph();
  auto v = m_view.slice(
      axis,
      compiler::Symbolic{g, Sym::Const(static_cast<int64_t>(index))},
      compiler::Symbolic{g, Sym::Const(1)});
  v = v.squeeze(axis);
  auto s = m_shape.squeeze(axis);
  return withView(std::move(s), std::move(v));
}

HostTensor HostTensor::narrow(std::size_t axis, std::uint64_t start,
                              std::uint64_t length) const {
  auto v = m_view.slice(
      axis,
      compiler::Symbolic{
          m_shape.graph(),
          Sym::Const(static_cast<std::int64_t>(start))},
      compiler::Symbolic{m_shape.graph(), Sym::Const(1)});
  auto sh = m_shape;
  sh[axis] = compiler::Symbolic{
      m_shape.graph(), Sym::Const(static_cast<std::int64_t>(length))};
  return withView(std::move(sh), std::move(v));
}

HostTensor
HostTensor::broadcastInDim(const TensorShape &to,
                           memory::span<const int64_t> axesMap) const {
  auto v = m_view.broadcastInDim(m_shape.dims(), to.dims(), axesMap);
  return withView(to, std::move(v));
}

HostTensor HostTensor::clone() const { return materializeContiguous(); }
bool HostTensor::sameStorageAs(const HostTensor &o) const {
  return m_store.get() == o.m_store.get();
}

HostTensor::HostTensor(TensorShape shape, TensorViewDesc view,
                       memory::shared_ptr<HostTensorStorage> storage)
    : m_shape(shape), m_view(std::move(view)), m_store(std::move(storage)) {}

HostTensor HostTensor::parse(const ::onnx::TensorProto &tensor,
                             const io::Path &externalDir) {
  static_assert(std::endian::native == std::endian::little,
                "vkcnn requires a little-endian host.");

  const std::string &name = tensor.name();

  if (tensor.has_segment()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" contains segment, not supported by vkcnn.",
        name));
  }

  // ---- dtype ----
  const auto dtypeOpt = Dtype::parse(tensor.data_type());
  if (!dtypeOpt) {
    throw std::runtime_error(
        fmt::format("vkcnn: Unsupported data type: \"{}\"",
                    Dtype::parse_to_string(tensor.data_type())));
  }
  const Dtype dt = *dtypeOpt;
  if (dt == Dtype::Undefined) {
    throw std::runtime_error(fmt::format(
        "vkcnn: initializer tensor (\"{}\") type cannot be undefined.", name));
  }

  std::vector<std::uint64_t> dims_u64;
  dims_u64.reserve(static_cast<size_t>(tensor.dims_size()));
  for (int i = 0; i < tensor.dims_size(); ++i) {
    const int64_t d = tensor.dims(i);
    if (d < 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: initializer has negative dim for tensor \"{}\"", name));
    }
    dims_u64.push_back(static_cast<std::uint64_t>(d));
  }
  TensorShape shape{nullptr, std::span<const std::uint64_t>(dims_u64)};

  // element count (rank-0 => 1)
  std::size_t n = 1;
  for (auto v : dims_u64)
    n *= static_cast<std::size_t>(v);

  // ---- helper: read external blob if requested ----
  auto read_external_blob =
      [&](std::size_t &outSize) -> std::vector<std::byte> {
    std::string location;
    std::size_t offset = 0, length = 0;
    for (int i = 0; i < tensor.external_data_size(); ++i) {
      const auto &kv = tensor.external_data(i);
      if (kv.key() == "location")
        location = kv.value();
      else if (kv.key() == "offset")
        offset = static_cast<std::size_t>(std::stoull(kv.value()));
      else if (kv.key() == "length")
        length = static_cast<std::size_t>(std::stoull(kv.value()));
    }
    if (location.empty()) {
      throw std::runtime_error("vkcnn: Missing external location");
    }
    if (externalDir.empty()) {
      throw std::runtime_error("vkcnn: ONNX contains external location, but no "
                               "external directory specified.");
    }

    auto p = externalDir / location;
    io::File f = io::File::open(p, io::File::OpenMode::Read);
    const std::size_t fileSize = f.size();
    if (offset > fileSize) {
      throw std::runtime_error(
          fmt::format("vkcnn: External file offset ({}) is past EOF of \"{}\"",
                      offset, p.str()));
    }
    const std::size_t toRead = (length == 0) ? (fileSize - offset) : length;
    memory::vector<std::byte> raw(toRead);
    f.read_exact(memory::span<std::byte>{raw.data(), toRead});
    outSize = toRead;
    return raw;
  };

  // ---- Build HostTensorStorage according to source precedence ----
  std::shared_ptr<HostTensorStorage> storage;

  auto make_from_raw = [&](const void *ptr, std::size_t bytes) {
    if (dt == Dtype::String) {
      // We only support strings via string_data(); raw/external-encoded strings
      // not supported.
      throw std::runtime_error(
          "vkcnn: string initializers must use string_data()");
    }
    const std::size_t elt = dt.size();
    if (elt == 0) {
      throw std::runtime_error("vkcnn: invalid dtype_size for this dtype");
    }
    if (bytes % elt != 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Invalid data length ({} bytes) not divisible by {}", bytes,
          elt));
    }
    const std::size_t count = bytes / elt;
    if (count != n) {
      throw std::runtime_error(
          fmt::format("vkcnn: Invalid data length. Expected {} elements ({} "
                      "bytes), got {} ({} bytes)",
                      n, n * elt, count, bytes));
    }
    storage = std::make_shared<HostTensorStorage>(
        HostTensorStorage::Raw(dt, ptr, bytes));
  };

  // 1) External file?
  if (tensor.data_location() == ::onnx::TensorProto_DataLocation_EXTERNAL) {
    std::size_t bytes = 0;
    auto raw = read_external_blob(bytes);
    make_from_raw(raw.data(), bytes);
  }
  // 2) raw_data?
  else if (!tensor.raw_data().empty()) {
    const auto &rd = tensor.raw_data();
    make_from_raw(rd.data(), rd.size());
  }
  // 3) typed fields
  else {
    switch (dt.kind()) {
    case DtypeKind::Sym:
      throw std::logic_error("vkcnn: Unexpected dtype.");
    case DtypeKind::Undefined:
      throw std::runtime_error(
          "vkcnn: Failed to parse tensor. undefined type!");
    case DtypeKind::Bool: {
      // ONNX puts bools in int32_data (0/1) commonly.
      const std::size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (n == 0) {
        storage = std::make_shared<HostTensorStorage>(
            HostTensorStorage::Raw(Dtype::Bool, nullptr, 0));
        break;
      }
      if (sz != n) {
        throw std::runtime_error("vkcnn: bool initializer length mismatch");
      }
      std::vector<std::uint8_t> bytes(n);
      for (size_t i = 0; i < n; ++i) {
        bytes[i] = tensor.int32_data(static_cast<int>(i)) ? std::uint8_t(1)
                                                          : std::uint8_t(0);
      }
      storage = std::make_shared<HostTensorStorage>(
          HostTensorStorage::Raw(Dtype::Bool, bytes.data(), bytes.size()));
      break;
    }
    case DtypeKind::Int8: {
      const size_t sz = static_cast<size_t>(tensor.int32_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT8 initializer");
      std::vector<std::int8_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data(static_cast<int>(i));
        if (x < std::numeric_limits<int8_t>::min() ||
            x > std::numeric_limits<int8_t>::max())
          throw std::runtime_error("vkcnn: INT8 value out of range");
        v[i] = static_cast<std::int8_t>(x);
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: INT8 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int8(v));
      break;
    }
    case DtypeKind::Int16: {
      const size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT16 initializer");
      std::vector<std::int16_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data(static_cast<int>(i));
        if (x < std::numeric_limits<int16_t>::min() ||
            x > std::numeric_limits<int16_t>::max())
          throw std::runtime_error("vkcnn: INT16 value out of range");
        v[i] = static_cast<std::int16_t>(x);
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: INT16 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Int16(v));
      break;
    }
    case DtypeKind::Int32: {
      const size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for INT32 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: INT32 initializer length mismatch");

      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int32(
          memory::span<const std::int32_t>(tensor.int32_data().data(), sz)));
      break;
    }
    case DtypeKind::Int64: {
      const size_t sz = static_cast<std::size_t>(tensor.int64_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int64_data for INT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: INT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Int64(
          std::span<const int64_t>(tensor.int64_data().data(), sz)));
      break;
    }
    case DtypeKind::Uint8: {
      const size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for UINT8 initializer");
      std::vector<std::uint8_t> v(sz);
      for (size_t i = 0; i < sz; ++i) {
        int32_t x = tensor.int32_data(static_cast<int>(i));
        if (x < 0 || x > std::numeric_limits<uint8_t>::max())
          throw std::runtime_error("vkcnn: UINT8 value out of range");
        v[i] = static_cast<std::uint8_t>(x);
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT8 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint8(v));
      break;
    }
    case DtypeKind::Uint16: {
      const std::size_t sz = static_cast<std::size_t>(tensor.int32_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing int32_data for UINT16 initializer");
      std::vector<std::uint16_t> v(sz);
      for (std::size_t i = 0; i < sz; ++i) {
        std::int32_t x = tensor.int32_data(static_cast<int>(i));
        if (x < 0 || x > std::numeric_limits<uint16_t>::max())
          throw std::runtime_error("vkcnn: UINT16 value out of range");
        v[i] = static_cast<std::uint16_t>(x);
      }
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT16 initializer length mismatch");
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint16(v));
      break;
    }
    case DtypeKind::Uint32: {
      // Primary: uint64_data carrier
      std::size_t sz64 = static_cast<std::size_t>(tensor.uint64_data_size());
      if (sz64) {
        if (sz64 != n)
          throw std::runtime_error("vkcnn: UINT32 initializer length mismatch");
        std::vector<std::uint32_t> v(sz64);
        for (std::size_t i = 0; i < sz64; ++i) {
          std::uint64_t x = tensor.uint64_data(static_cast<int>(i));
          if (x > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("vkcnn: UINT32 value out of range");
          v[i] = static_cast<std::uint32_t>(x);
        }
        storage =
            std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(v));
        break;
      }
      // Fallback: some exporters use int32_data for UINT32
      size_t sz32 = static_cast<std::size_t>(tensor.int32_data_size());
      if (sz32 == 0 && n != 0)
        throw std::runtime_error("vkcnn: missing data for UINT32 initializer");
      if (sz32 != n)
        throw std::runtime_error("vkcnn: UINT32 initializer length mismatch");
      std::vector<std::uint32_t> v(sz32);
      for (size_t i = 0; i < sz32; ++i) {
        int32_t x = tensor.int32_data(static_cast<int>(i));
        if (x < 0)
          throw std::runtime_error(
              "vkcnn: negative value in UINT32 initializer");
        v[i] = static_cast<std::uint32_t>(x);
      }
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::Uint32(v));
      break;
    }
    case DtypeKind::Uint64: {
      const size_t sz = static_cast<std::size_t>(tensor.uint64_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing uint64_data for UINT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: UINT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::Uint64(
          std::span<const uint64_t>(tensor.uint64_data().data(), sz)));
      break;
    }
    case DtypeKind::Float32: {
      const size_t sz = static_cast<std::size_t>(tensor.float_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing float_data for FLOAT32 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: FLOAT32 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::F32(
          std::span<const float>(tensor.float_data().data(), sz)));
      break;
    }
    case DtypeKind::Float64: {
      const size_t sz = static_cast<std::size_t>(tensor.double_data_size());
      if (sz == 0 && n != 0)
        throw std::runtime_error(
            "vkcnn: missing double_data for FLOAT64 initializer");
      if (sz != n)
        throw std::runtime_error("vkcnn: FLOAT64 initializer length mismatch");
      storage = std::make_shared<HostTensorStorage>(HostTensorStorage::F64(
          std::span<const double>(tensor.double_data().data(), sz)));
      break;
    }
    case DtypeKind::Float16: {
      // Usually provided via raw_data; if not present, we consider it missing.
      throw std::runtime_error(
          "vkcnn: FLOAT16 initializer must use raw_data or external_data");
    }
    case DtypeKind::String: {
      const size_t sz = static_cast<std::size_t>(tensor.string_data_size());
      if (n == 0) {
        storage = std::make_shared<HostTensorStorage>(
            HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0));
        break;
      }
      if (sz != n) {
        throw std::runtime_error("vkcnn: STRING initializer length mismatch");
      }
      // Build vector<string> then storage
      std::vector<std::string> vals(sz);
      for (size_t i = 0; i < sz; ++i) {
        vals[i] = tensor.string_data(static_cast<int>(i));
      }
      storage =
          std::make_shared<HostTensorStorage>(HostTensorStorage::String(vals));
      break;
    }
    default:
      throw std::runtime_error("vkcnn: unsupported initializer dtype");
    }
  }
  // ---- HostTensor (identity view over the shape) ----
  return HostTensor(shape, std::move(storage));
}

std::int64_t HostTensor::loadI64(memory::span<const std::uint64_t> idx) const {
  const std::size_t k = m_view.constIndexOf(idx);
  switch (m_store->type().kind()) {
  case DtypeKind::Int8:
    return static_cast<int64_t>(
        static_cast<const int8_t *>(m_store->data())[k]);
  case DtypeKind::Int16:
    return static_cast<int64_t>(
        static_cast<const int16_t *>(m_store->data())[k]);
  case DtypeKind::Int32:
    return static_cast<int64_t>(
        static_cast<const int32_t *>(m_store->data())[k]);
  case DtypeKind::Int64:
    return static_cast<int64_t>(
        static_cast<const int64_t *>(m_store->data())[k]);
  case DtypeKind::Uint8:
    return static_cast<int64_t>(
        static_cast<const uint8_t *>(m_store->data())[k]);
  case DtypeKind::Uint16:
    return static_cast<int64_t>(
        static_cast<const uint16_t *>(m_store->data())[k]);
  case DtypeKind::Uint32:
    return static_cast<int64_t>(
        static_cast<const uint32_t *>(m_store->data())[k]);
  case DtypeKind::Uint64:
    return static_cast<int64_t>(
        static_cast<const uint64_t *>(m_store->data())[k]);
  case DtypeKind::Undefined:
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    throw std::logic_error("load_int64: non-integer dtype");
  }
  diag::unreachable();
}
std::uint64_t HostTensor::loadU64(memory::span<const std::uint64_t> idx) const {
  const std::size_t k = m_view.constIndexOf(idx);
  switch (m_store->type().kind()) {
  case DtypeKind::Int8:
    return static_cast<uint64_t>(
        static_cast<const int8_t *>(m_store->data())[k]);
  case DtypeKind::Int16:
    return static_cast<uint64_t>(
        static_cast<const int16_t *>(m_store->data())[k]);
  case DtypeKind::Int32:
    return static_cast<uint64_t>(
        static_cast<const int32_t *>(m_store->data())[k]);
  case DtypeKind::Int64:
    return static_cast<uint64_t>(
        static_cast<const int64_t *>(m_store->data())[k]);
  case DtypeKind::Uint8:
    return static_cast<uint64_t>(
        static_cast<const uint8_t *>(m_store->data())[k]);
  case DtypeKind::Uint16:
    return static_cast<uint64_t>(
        static_cast<const uint16_t *>(m_store->data())[k]);
  case DtypeKind::Uint32:
    return static_cast<uint64_t>(
        static_cast<const uint32_t *>(m_store->data())[k]);
  case DtypeKind::Uint64:
    return static_cast<uint64_t>(
        static_cast<const uint64_t *>(m_store->data())[k]);
  case DtypeKind::Undefined:
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    throw std::logic_error("load_uint64: non-integer dtype");
  }
  diag::unreachable();
}
double HostTensor::loadDouble(memory::span<const std::uint64_t> idx) const {
  const std::size_t k = m_view.constIndexOf(idx);
  switch (m_store->type().kind()) {
  case DtypeKind::Float16:
    return static_cast<double>(
        static_cast<const memory::f16 *>(m_store->data())[k]);
  case DtypeKind::Float32:
    return static_cast<double>(static_cast<const float *>(m_store->data())[k]);
  case DtypeKind::Float64:
    return static_cast<const double *>(m_store->data())[k];
  case DtypeKind::Undefined:
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    throw std::logic_error("load_double: unsupported dtype");
  }
  diag::unreachable();
}
Sym HostTensor::loadSym(memory::span<const std::uint64_t> idx) const {
  const std::size_t k = m_view.constIndexOf(idx);
  switch (m_store->type().kind()) {
  case DtypeKind::Sym:
    return static_cast<const Sym *>(m_store->data())[k];
  case DtypeKind::Int8:
    return Sym::Const(
        static_cast<int64_t>(static_cast<const int8_t *>(m_store->data())[k]));
  case DtypeKind::Int16:
    return Sym::Const(
        static_cast<int64_t>(static_cast<const int16_t *>(m_store->data())[k]));
  case DtypeKind::Int32:
    return Sym::Const(
        static_cast<int64_t>(static_cast<const int32_t *>(m_store->data())[k]));
  case DtypeKind::Int64:
    return Sym::Const(
        static_cast<const int64_t *>(m_store->data())[k]);
  case DtypeKind::Uint8:
    return Sym::Const(
        static_cast<int64_t>(static_cast<const uint8_t *>(m_store->data())[k]));
  case DtypeKind::Uint16:
    return Sym::Const(static_cast<int64_t>(
        static_cast<const uint16_t *>(m_store->data())[k]));
  case DtypeKind::Uint32:
    return Sym::Const(static_cast<int64_t>(
        static_cast<const uint32_t *>(m_store->data())[k]));
  case DtypeKind::Uint64:
    return Sym::Const(static_cast<int64_t>(
        static_cast<const uint64_t *>(m_store->data())[k]));
  case DtypeKind::Float16:
  case DtypeKind::Float32:
  case DtypeKind::Float64:
  case DtypeKind::Undefined:
  case DtypeKind::String:
  case DtypeKind::Bool:
    throw std::logic_error("load_sym: invalid dtype");
  }
  diag::unreachable();
}
memory::span<const float> HostTensor::floats() const {
  assert(isContiguous());
  assert(sizeof(float) == elemSize());
  const auto *p = static_cast<const float *>(data());
  // count:
  std::size_t n = 1;
  for (auto d : m_shape.toU64())
    n *= static_cast<std::size_t>(d);
  return {p, n};
}
} // namespace denox::onnx::details
