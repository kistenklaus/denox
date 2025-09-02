#pragma once

#include "vkcnn/common/model/Model.hpp"
#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include <cstdlib>
#include <cstring>
#include <fmt/base.h>
#include <stdexcept>
#include <variant>
namespace vkcnn::details {

class TensorShape {
public:
  // ---- ctors (graph is required) ----
  TensorShape(std::shared_ptr<SymGraph> g, std::vector<Symbolic> dims)
      : m_graph(std::move(g)), m_dims(std::move(dims)) {
    assert(m_graph && "TensorShape: graph must not be null");
    // Validate dims belong to the same graph in debug
    for (const auto &d : m_dims) {
      assert(!d.graph() || d.graph().get() == m_graph.get());
      if (d.isConstant()) {
        assert(d.constant() >= 0);
      }
    }
  }

  TensorShape(const std::shared_ptr<SymGraph> &g, std::span<const Sym> dims)
      : m_graph(g) {
    assert(m_graph && "TensorShape: graph must not be null");
    m_dims.reserve(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      Symbolic s{m_graph, dims[i]};
      if (s.isConstant()) {
        assert(s.constant() >= 0);
      }
      m_dims.emplace_back(std::move(s));
    }
  }

  TensorShape(const std::shared_ptr<SymGraph> &g,
              std::span<const std::uint64_t> dims)
      : m_graph(g) {
    assert(m_graph && "TensorShape: graph must not be null");
    m_dims.reserve(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      m_dims.emplace_back(m_graph, Sym::Const(static_cast<int64_t>(dims[i])));
    }
  }

  TensorShape(const std::shared_ptr<SymGraph> &g,
              std::span<const std::int64_t> dims)
      : m_graph(g) {
    assert(m_graph && "TensorShape: graph must not be null");
    m_dims.reserve(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      assert(dims[i] >= 0);
      m_dims.emplace_back(m_graph, Sym::Const(dims[i]));
    }
  }

  // ---- queries ----
  std::span<const Symbolic> dims() const { return m_dims; }
  size_t rank() const { return m_dims.size(); }
  bool empty() const { return m_dims.empty(); }

  bool isConstant() const {
    for (const auto &d : m_dims)
      if (d.isSymbolic())
        return false;
    return true;
  }
  bool hasSymbolic() const { return !isConstant(); }

  bool hasZeroDim() const {
    for (const auto &d : m_dims)
      if (d.isConstant() && d.constant() == 0)
        return true;
    return false;
  }

  Symbolic numel() const {
    Symbolic one{m_graph, Sym::Const(1)};
    Symbolic n = one;
    for (const auto &d : m_dims) {
      if (d.isConstant() && d.constant() == 0)
        return Symbolic{m_graph, Sym::Const(0)};
      n = n * d;
    }
    return n;
  }

  std::vector<std::uint64_t> toU64() const {
    assert(isConstant());
    std::vector<std::uint64_t> out;
    out.reserve(m_dims.size());
    for (const auto &d : m_dims) {
      assert(d.isConstant());
      out.push_back(static_cast<std::uint64_t>(d.constant()));
    }
    return out;
  }

  TensorShape permute(std::span<const int64_t> perm) const {
    assert(perm.size() == m_dims.size());
    assert(isPermutation(perm, m_dims.size()));
    std::vector<Symbolic> v(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      const size_t p = static_cast<size_t>(perm[i]);
      v[i] = m_dims[p];
    }
    return TensorShape{m_graph, std::move(v)};
  }

  TensorShape unsqueeze(size_t axis) const {
    assert(axis <= m_dims.size());
    std::vector<Symbolic> v = m_dims;
    v.insert(v.begin() + axis, Symbolic{m_graph, Sym::Const(1)});
    return TensorShape{m_graph, std::move(v)};
  }

  TensorShape squeeze(size_t axis) const {
    assert(axis < m_dims.size());
    std::vector<Symbolic> v = m_dims;
    v.erase(v.begin() + axis);
    return TensorShape{m_graph, std::move(v)};
  }

  static TensorShape broadcast(const TensorShape &a, const TensorShape &b) {
    assert(a.m_graph && b.m_graph && a.m_graph.get() == b.m_graph.get());
    auto g = a.m_graph;

    const size_t ra = a.rank(), rb = b.rank();
    const size_t r = (ra > rb ? ra : rb);
    std::vector<Symbolic> out(r, Symbolic{g, Sym::Const(1)});

    for (size_t i = 0; i < r; ++i) {
      const bool haveA = (i < ra);
      const bool haveB = (i < rb);
      const Symbolic da =
          haveA ? a.m_dims[ra - 1 - i] : Symbolic{g, Sym::Const(1)};
      const Symbolic db =
          haveB ? b.m_dims[rb - 1 - i] : Symbolic{g, Sym::Const(1)};

      if (da.isConstant() && da.constant() == 1) {
        out[r - 1 - i] = db;
      } else if (db.isConstant() && db.constant() == 1) {
        out[r - 1 - i] = da;
      } else {
        if (da == db)
          out[r - 1 - i] = da;
        else
          throw std::runtime_error("broadcast: incompatible dimensions");
      }
    }
    return TensorShape{g, std::move(out)};
  }

  const Symbolic &operator[](size_t i) const { return m_dims[i]; }
  Symbolic &operator[](size_t i) { return m_dims[i]; }

  std::shared_ptr<SymGraph> graph() const { return m_graph; }

private:
  static bool isPermutation(std::span<const int64_t> p, size_t r) {
    if (p.size() != r)
      return false;
    std::vector<uint8_t> seen(r, 0);
    for (auto v : p) {
      if (v < 0)
        return false;
      const size_t u = static_cast<size_t>(v);
      if (u >= r)
        return false;
      if (seen[u]++)
        return false;
    }
    return true;
  }

  std::shared_ptr<SymGraph> m_graph; // never null after construction
  std::vector<Symbolic> m_dims;      // non-negative extents (symbolic or const)
};

class TensorViewDesc {
public:
  static TensorViewDesc Identity(const TensorShape &shape) {
    auto graph = shape.graph();
    assert(graph);
    std::size_t r = shape.rank();
    std::vector<Symbolic> strides(r);
    Symbolic offset{graph, Sym::Const(0)};
    if (r != 0) {
      strides[r - 1] = Symbolic{graph, Sym::Const(1)};
      for (std::size_t d = r - 1; d-- > 0;) {
        strides[d] = strides[d + 1] * shape.dims()[d + 1];
      }
    }
    return TensorViewDesc{std::move(strides), std::move(offset)};
  }

  std::span<const Symbolic> strides() const { return m_strides; }
  const Symbolic &offset() const { return m_offset; }

  bool isConstant() const {
    if (m_offset.isSymbolic()) {
      return false;
    }
    for (const auto &s : m_strides) {
      if (s.isSymbolic()) {
        return false;
      }
    }
    return true;
  }

  std::size_t constIndexOf(std::span<const std::uint64_t> idx) const {
    assert(isConstant());
    assert(idx.size() == m_strides.size());

    if (m_strides.empty()) {
      const std::int64_t off = static_cast<std::int64_t>(m_offset.constant());
      assert(off >= 0);
      return static_cast<std::size_t>(off);
    }
    std::int64_t acc = static_cast<std::int64_t>(m_offset.constant());
    for (std::size_t i = 0; i < idx.size(); ++i) {
      const std::int64_t si =
          static_cast<std::int64_t>(m_strides[i].constant());
      acc += si * static_cast<std::int64_t>(idx[i]);
    }
    assert(acc >= 0);
    return static_cast<std::size_t>(acc);
  }

  Symbolic indexOfSym(std::span<const Symbolic> idx) const {
    assert(idx.size() == m_strides.size());
    Symbolic acc = m_offset;
    for (std::size_t i = 0; i < idx.size(); ++i) {
      acc = acc + (m_strides[i] * idx[i]);
    }
    return acc;
  }

  Symbolic symIndexOf(std::span<const std::uint64_t> idx) const {
    assert(idx.size() == m_strides.size());
    Symbolic acc = m_offset;
    for (std::size_t i = 0; i < idx.size(); ++i) {
      acc = acc + (m_strides[i] * static_cast<std::int64_t>(idx[i]));
    }
    return acc;
  }

  TensorViewDesc withOffset(const Symbolic &delta) const {
    return TensorViewDesc{m_strides, m_offset + delta};
  }

  TensorViewDesc permute(std::span<const int64_t> perm) const {
    assert(isPermutation(perm, m_strides.size()));
    std::vector<Symbolic> s(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      const auto p = static_cast<size_t>(perm[i]);
      assert(p < m_strides.size());
      s[i] = m_strides[p];
    }
    return TensorViewDesc{std::move(s), m_offset};
  }

  TensorViewDesc slice(size_t axis, const Symbolic &start,
                       const Symbolic &step) const {
    assert(axis < m_strides.size());
    if (step.isConstant()) {
      assert(step.constant() != 0 && "step must be nonzero");
    }
    auto s = m_strides;
    Symbolic off = m_offset + s[axis] * start;
    s[axis] = s[axis] * step; // may be negative → allowed
    return TensorViewDesc{std::move(s), std::move(off)};
  }

  TensorViewDesc reverse(size_t axis, const Symbolic &size) const {
    assert(axis < m_strides.size());
    auto s = m_strides;
    Symbolic off = m_offset;
    if (!(size.isConstant() && size.constant() == 0)) {
      off = off + s[axis] * (size - 1);
      s[axis] = s[axis] * static_cast<int64_t>(-1);
    }
    return TensorViewDesc{std::move(s), std::move(off)};
  }

  TensorViewDesc
  normalizeNegativeStrides(std::span<const Symbolic> shape) const {
    assert(shape.size() == m_strides.size());
    auto s = m_strides;
    Symbolic off = m_offset;
    for (size_t i = 0; i < s.size(); ++i) {
      // if stride < 0, move offset to front and flip stride
      // NOTE: works when stride sign is known (constant). If symbolic sign,
      // skip.
      if (s[i].isConstant() && s[i].constant() < 0) {
        off = off + s[i] * (shape[i] - 1); // s[i] is negative here
        s[i] = s[i] * static_cast<int64_t>(-1);
      }
    }
    return TensorViewDesc{std::move(s), std::move(off)};
  }

  TensorViewDesc unsqueeze(size_t axis) const {
    auto s = m_strides;
    const size_t r = s.size();
    assert(axis <= r);
    Symbolic one{m_offset.graph(), Sym::Const(1)};
    Symbolic stride_here = (r == 0) ? one : (axis == r ? one : s[axis]);
    s.insert(s.begin() + axis, stride_here);
    return TensorViewDesc{std::move(s), m_offset};
  }

  TensorViewDesc squeeze(size_t axis) const {
    auto s = m_strides;
    assert(axis < s.size());
    s.erase(s.begin() + axis);
    return TensorViewDesc{std::move(s), m_offset};
  }

  TensorViewDesc broadcastInDim(std::span<const Symbolic> fromShape,
                                std::span<const Symbolic> toShape,
                                std::span<const int64_t> axesMap) const {
    assert(axesMap.size() == fromShape.size());
    std::vector<Symbolic> s(toShape.size(),
                            Symbolic{m_offset.graph(), Sym::Const(0)});
    for (size_t i = 0; i < axesMap.size(); ++i) {
      size_t ax = (size_t)axesMap[i];
      assert(ax < s.size());
      // If fromShape[i]==1 and toShape[ax]>1 → stride 0; else copy stride
      if (fromShape[i].isConstant() && toShape[ax].isConstant()) {
        if (fromShape[i].constant() == 1 && toShape[ax].constant() > 1) {
          s[ax] = Symbolic{m_offset.graph(), Sym::Const(0)};
        } else {
          s[ax] = m_strides[i];
        }
      } else {
        // symbolic sizes: conservatively carry the stride; caller should ensure
        // correctness
        s[ax] = m_strides[i];
      }
    }
    return TensorViewDesc{std::move(s), m_offset};
  }

  bool isRowMajorContiguous(std::span<const uint64_t> dims) const {
    if (dims.size() != m_strides.size())
      return false;
    if (dims.empty())
      return true;
    // all strides must be constant
    for (const auto &st : m_strides)
      if (st.isSymbolic())
        return false;
    int64_t expect = 1;
    for (size_t i = dims.size(); i-- > 0;) {
      const int64_t si = static_cast<int64_t>(m_strides[i].constant());
      if (si != expect)
        return false;
      expect *= static_cast<int64_t>(dims[i]);
    }
    return true;
  }

  bool hasNegativeStride() const {
    for (const auto &st : m_strides) {
      if (st.isConstant() && st.constant() < 0)
        return true;
    }
    return false;
  }

  std::size_t rank() const { return m_strides.size(); }

private:
  static bool isPermutation(std::span<const int64_t> p, size_t r) {
    std::vector<uint8_t> seen(r, 0);
    for (auto v : p) {
      if (v < 0)
        return false;
      size_t u = static_cast<size_t>(v);
      if (u >= r)
        return false;
      if (seen[u]++)
        return false; // dup
    }
    return p.size() == r; // redundant with your outer assert, but safe
  }

  explicit TensorViewDesc(std::vector<Symbolic> strides, Symbolic offset)
      : m_strides(std::move(strides)), m_offset(std::move(offset)) {}
  std::vector<Symbolic> m_strides;
  Symbolic m_offset;
};

class HostTensorStorage {
public:
  HostTensorStorage()
      : m_type(Dtype::Undefined), m_raw(nullptr), m_byteSize(0) {}
  HostTensorStorage(const HostTensorStorage &) = delete;
  HostTensorStorage &operator=(const HostTensorStorage &) = delete;
  HostTensorStorage(HostTensorStorage &&o)
      : m_type(std::exchange(o.m_type, Dtype::Undefined)),
        m_raw(std::exchange(o.m_raw, nullptr)),
        m_byteSize(std::exchange(o.m_byteSize, 0)) {}
  HostTensorStorage &operator=(HostTensorStorage &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    std::swap(m_type, o.m_type);
    std::swap(m_raw, o.m_raw);
    std::swap(m_byteSize, o.m_byteSize);
    return *this;
  }
  ~HostTensorStorage() { release(); }
  void release() {
    if (!m_raw)
      return;
    if (m_type == Dtype::String) {
      const std::size_t n = m_byteSize / sizeof(char *); // not char**
      char **arr = static_cast<char **>(m_raw);
      for (std::size_t i = 0; i < n; ++i)
        free(arr[i]);
    }
    free(m_raw);
    m_raw = nullptr;
    m_byteSize = 0;
    m_type = Dtype::Undefined;
  }

  static HostTensorStorage TakeOwnership(Dtype type, void *raw,
                                         std::size_t bytes) {
    return HostTensorStorage{type, raw,
                             bytes}; // private ctor already does this
  }
  static HostTensorStorage Raw(Dtype type, const void *raw,
                               std::size_t byteSize) {
    assert(type != Dtype::String);
    void *ptr = malloc(byteSize);
    std::memcpy(ptr, raw, byteSize);
    return HostTensorStorage{type, ptr, byteSize};
  }
  static HostTensorStorage Bool(std::span<const bool> values) {
    static_assert(sizeof(std::uint8_t) == 1);
    std::size_t rawSize = values.size() * dtype_size(Dtype::Bool);
    void *raw = malloc(rawSize);
    if (dtype_size(Dtype::Bool) == sizeof(bool)) {
      std::memcpy(raw, values.data(), values.size_bytes());
    } else {
      for (std::size_t i = 0; i < values.size(); ++i) {
        static_cast<std::uint8_t *>(raw)[i] =
            values[i] ? std::uint8_t(1) : std::uint8_t(0);
      }
    }
    return HostTensorStorage{Dtype::Bool, raw, rawSize};
  }
  static HostTensorStorage Int8(std::span<const std::int8_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Int8, raw, values.size_bytes()};
  }
  static HostTensorStorage Int16(std::span<const std::int16_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Int16, raw, values.size_bytes()};
  }
  static HostTensorStorage Int32(std::span<const std::int32_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Int32, raw, values.size_bytes()};
  }
  static HostTensorStorage Int64(std::span<const std::int64_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Int64, raw, values.size_bytes()};
  }
  static HostTensorStorage Uint8(std::span<const std::uint8_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Uint8, raw, values.size_bytes()};
  }
  static HostTensorStorage Uint16(std::span<const std::uint16_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Uint16, raw, values.size_bytes()};
  }
  static HostTensorStorage Uint32(std::span<const std::uint32_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Uint32, raw, values.size_bytes()};
  }
  static HostTensorStorage Uint64(std::span<const std::uint64_t> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Uint64, raw, values.size_bytes()};
  }
  static HostTensorStorage Sym(std::span<const Sym> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Sym, raw, values.size_bytes()};
  }
  static HostTensorStorage F16(std::span<const f16> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Float16, raw, values.size_bytes()};
  }
  static HostTensorStorage F32(std::span<const f32> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Float32, raw, values.size_bytes()};
  }
  static HostTensorStorage F64(std::span<const f64> values) {
    void *raw = malloc(values.size_bytes());
    std::memcpy(raw, values.data(), values.size_bytes());
    return HostTensorStorage{Dtype::Float64, raw, values.size_bytes()};
  }
  static HostTensorStorage String(std::span<const std::string> values) {
    std::size_t rawSize = sizeof(char *) * values.size();
    void *raw = malloc(rawSize);
    for (std::size_t i = 0; i < values.size(); ++i) {
      char *str = static_cast<char *>(malloc(values[i].size() + 1));
      std::memcpy(str, values[i].data(), values[i].size());
      str[values[i].size()] = '\0';
      static_cast<char **>(raw)[i] = str;
    }
    return HostTensorStorage{Dtype::String, raw, rawSize};
  }

  std::span<const std::uint8_t> boolean() const {
    assert(m_type == Dtype::Bool);
    return {static_cast<const std::uint8_t *>(m_raw), m_byteSize};
  }
  std::span<const std::int8_t> i8() const {
    assert(m_type == Dtype::Int8);
    return {static_cast<const std::int8_t *>(m_raw), m_byteSize};
  }
  std::span<const std::int16_t> i16() const {
    assert(m_type == Dtype::Int16);
    return {static_cast<const std::int16_t *>(m_raw),
            m_byteSize / sizeof(std::int16_t)};
  }
  std::span<const std::int32_t> i32() const {
    assert(m_type == Dtype::Int32);
    return {static_cast<const std::int32_t *>(m_raw),
            m_byteSize / sizeof(std::int32_t)};
  }
  std::span<const std::int64_t> i64() const {
    assert(m_type == Dtype::Int64);
    return {static_cast<const std::int64_t *>(m_raw),
            m_byteSize / sizeof(std::int64_t)};
  }
  std::span<const std::uint8_t> u8() const {
    assert(m_type == Dtype::Uint8);
    return {static_cast<const std::uint8_t *>(m_raw), m_byteSize};
  }
  std::span<const std::uint16_t> u16() const {
    assert(m_type == Dtype::Uint16);
    return {static_cast<const std::uint16_t *>(m_raw),
            m_byteSize / sizeof(std::uint16_t)};
  }
  std::span<const std::uint32_t> u32() const {
    assert(m_type == Dtype::Uint32);
    return {static_cast<const std::uint32_t *>(m_raw),
            m_byteSize / sizeof(std::uint32_t)};
  }
  std::span<const std::uint64_t> u64() const {
    assert(m_type == Dtype::Uint64);
    return {static_cast<const std::uint64_t *>(m_raw),
            m_byteSize / sizeof(std::uint64_t)};
  }
  std::span<const vkcnn::Sym> sym() const {
    assert(m_type == Dtype::Sym);
    return {static_cast<const vkcnn::Sym *>(m_raw),
            m_byteSize / sizeof(vkcnn::Sym)};
  }
  std::span<const vkcnn::f16> f16() const {
    assert(m_type == Dtype::Float16);
    return {static_cast<const vkcnn::f16 *>(m_raw),
            m_byteSize / sizeof(vkcnn::f16)};
  }
  std::span<const vkcnn::f32> f32() const {
    assert(m_type == Dtype::Float32);
    return {static_cast<const vkcnn::f32 *>(m_raw),
            m_byteSize / sizeof(vkcnn::f32)};
  }
  std::span<const vkcnn::f64> f64() const {
    assert(m_type == Dtype::Float64);
    return {static_cast<const vkcnn::f64 *>(m_raw),
            m_byteSize / sizeof(vkcnn::f64)};
  }
  std::span<const char *> strs() const {
    return {static_cast<const char **>(m_raw), m_byteSize / sizeof(char *)};
  }

  const void *data() const { return m_raw; }
  void *data() { return m_raw; }

  Dtype type() const { return m_type; }

private:
  explicit HostTensorStorage(Dtype dtype, void *raw, std::size_t byteSize)
      : m_type(dtype), m_raw(raw), m_byteSize(byteSize) {}
  Dtype m_type;
  void *m_raw;
  std::size_t m_byteSize;
};

class HostTensor {
public:
  explicit HostTensor(TensorShape shape,
                      std::shared_ptr<HostTensorStorage> storage)
      : m_shape(shape), m_view(TensorViewDesc::Identity(shape)),
        m_store(std::move(storage)) {}

  TensorShape shape() const { return m_shape; }
  TensorViewDesc view() const { return m_view; }
  Dtype type() const { return m_store->type(); }
  const std::shared_ptr<HostTensorStorage> &storage() const { return m_store; }

  size_t rank() const { return m_shape.rank(); }
  bool isConstant() const { return m_shape.isConstant(); }
  Symbolic numel() const { return m_shape.numel(); }
  std::size_t elemSize() const { return dtype_size(type()); }

  bool isContiguous() const {
    if (!m_shape.isConstant())
      return false;
    return m_view.isRowMajorContiguous(m_shape.toU64());
  }

  std::size_t sizeBytesIfStatic() const {
    if (!m_shape.isConstant())
      return 0;                   // unknown
    auto elems = m_shape.toU64(); // may overflow, you said that's ok
    std::size_t n = 1;
    for (auto e : elems)
      n *= static_cast<std::size_t>(e);
    return n * elemSize();
  }

  std::size_t byteOffset() const {
    assert(m_view.isConstant());
    return static_cast<std::size_t>(m_view.offset().constant()) * elemSize();
  }

  const void *data() const {
    return static_cast<const std::byte *>(m_store->data()) + byteOffset();
  }

  void *data() {
    return static_cast<std::byte *>(m_store->data()) + byteOffset();
  }

  template <class T> std::span<const T> span() const {
    assert(isContiguous());
    assert(sizeof(T) == elemSize());
    const auto *p = static_cast<const T *>(data());
    // count:
    std::size_t n = 1;
    for (auto d : m_shape.toU64())
      n *= static_cast<std::size_t>(d);
    return {p, n};
  }

  HostTensor withView(TensorShape newShape, TensorViewDesc newView) const {
    return HostTensor(std::move(newShape), std::move(newView), m_store);
  }

  HostTensor permute(std::span<const int64_t> perm) const {
    auto newShape = m_shape.permute(perm);
    auto newView = m_view.permute(perm);
    return withView(std::move(newShape), std::move(newView));
  }

  HostTensor unsqueeze(size_t axis) const {
    auto newShape = m_shape.unsqueeze(axis);
    auto newView = m_view.unsqueeze(axis);
    return withView(std::move(newShape), std::move(newView));
  }

  HostTensor squeeze(size_t axis) const {
    auto newShape = m_shape.squeeze(axis);
    auto newView = m_view.squeeze(axis);
    return withView(std::move(newShape), std::move(newView));
  }

  HostTensor materializeContiguous() const {
    // 1) Static shape required
    if (!m_shape.isConstant()) {
      throw std::runtime_error("materializeContiguous: shape must be static");
    }

    const std::size_t elem = elemSize();
    const auto sizes = m_shape.toU64(); // rank dims
    const size_t rank = sizes.size();

    // total elements
    std::size_t totalElems = 1;
    for (auto s : sizes)
      totalElems *= static_cast<std::size_t>(s);
    const std::size_t bytes = totalElems * elem;

    // 2) Zero-size: build empty storage of same dtype
    if (bytes == 0) {
      std::shared_ptr<HostTensorStorage> empty;
      if (type() == Dtype::String) {
        empty = std::make_shared<HostTensorStorage>(
            HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0));
      } else {
        empty = std::make_shared<HostTensorStorage>(
            HostTensorStorage::Raw(type(), nullptr, 0));
      }
      return HostTensor(m_shape, std::move(empty));
    }

    // 3) Already base-contiguous with zero offset? reuse
    if (type() != Dtype::String && isContiguous() &&
        m_view.offset().isConstant() && m_view.offset().constant() == 0) {
      return *this;
    }

    // 4) Build constant strides (in BYTES) and base offset (in BYTES)
    const auto vstr = m_view.strides(); // Symbolic
    std::vector<long long> strideBytes(vstr.size());
    for (size_t i = 0; i < vstr.size(); ++i) {
      assert(vstr[i].isConstant() &&
             "materializeContiguous: view must be constant for now");
      const long long si = static_cast<long long>(vstr[i].constant());
      strideBytes[i] = si * static_cast<long long>(elem);
    }
    assert(m_view.offset().isConstant());
    const long long base = static_cast<long long>(m_view.offset().constant()) *
                           static_cast<long long>(elem);

    // Common N-D iteration helpers
    const auto *srcBase = static_cast<const std::byte *>(m_store->data());

    auto advance_index = [&](std::vector<std::size_t> &idx) -> bool {
      // returns false when finished (wrapped around)
      if (idx.empty())
        return false; // rank-0 handled separately
      size_t ax = idx.size();
      while (ax > 0) {
        --ax;
        if (++idx[ax] < sizes[ax])
          return true;
        idx[ax] = 0;
      }
      return false; // wrapped all the way → done
    };

    auto compute_offset_bytes =
        [&](std::span<const std::size_t> idx) -> long long {
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
        const auto *table = reinterpret_cast<char *const *>(srcBase);
        // offBytes is a multiple of sizeof(char*) by construction
        const std::size_t idxElem = static_cast<std::size_t>(
            offBytes / static_cast<long long>(sizeof(char *)));
        const char *srcp = table[idxElem];
        const std::size_t len = std::strlen(srcp);
        char *cp = static_cast<char *>(std::malloc(len + 1));
        if (!cp)
          throw std::bad_alloc();
        std::memcpy(cp, srcp, len + 1); // includes '\0'
        outTable[k++] = cp;
      };

      if (rank == 0) {
        // single string scalar
        copy_one(base);
      } else {
        std::vector<std::size_t> idx(rank, 0);
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

    // 6) Non-string: memcpy fast paths
    void *dst = std::malloc(bytes);
    if (!dst)
      throw std::bad_alloc();

    auto copy_elementwise = [&]() {
      if (rank == 0) {
        std::memcpy(dst, srcBase + base, elem);
        return;
      }
      std::vector<std::size_t> idx(rank, 0);
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
      // If innermost axis has stride == elem, copy full rows with one memcpy.
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

      std::vector<std::size_t> idx(rank - 1, 0);
      std::byte *out = static_cast<std::byte *>(dst);

      while (true) {
        long long off = base;
        for (size_t ax = 0; ax < rank - 1; ++ax) {
          off += strideBytes[ax] * static_cast<long long>(idx[ax]);
        }
        const std::byte *src = srcBase + off; // start of the row
        std::memcpy(out, src, rowBytes);
        out += rowBytes;

        // advance idx over the outer (rank-1) axes
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

  std::size_t sizeElemsIfStatic() const {
    if (!m_shape.isConstant())
      return 0;
    std::size_t n = 1;
    for (auto d : m_shape.toU64())
      n *= static_cast<std::size_t>(d);
    return n;
  }

  HostTensor contiguous() const {
    return (isContiguous() && m_view.offset().isConstant() &&
            m_view.offset().constant() == 0)
               ? *this
               : materializeContiguous();
  }

  HostTensor reshape(const TensorShape &newShape) const {
    // require static sizes & same numel
    assert(isConstant() && "reshape needs static size");
    auto oldN = sizeElemsIfStatic();
    std::size_t newN = 1;
    for (auto d : newShape.toU64())
      newN *= (size_t)d;
    if (oldN != newN)
      throw std::runtime_error("reshape: numel mismatch");

    if (!isContiguous() ||
        !(m_view.offset().isConstant() && m_view.offset().constant() == 0)) {
      return materializeContiguous().reshape(newShape);
    }
    auto newView = TensorViewDesc::Identity(newShape);
    return withView(newShape, std::move(newView));
  }

  HostTensor select(size_t axis, uint64_t index) const {
    auto g = m_shape.graph();
    // move the offset to the selected slice
    auto v =
        m_view.slice(axis, Symbolic{g, Sym::Const(static_cast<int64_t>(index))},
                     Symbolic{g, Sym::Const(1)});
    // reduce rank in both shape and view
    v = v.squeeze(axis);
    auto s = m_shape.squeeze(axis);
    return withView(std::move(s), std::move(v));
  }

  HostTensor narrow(size_t axis, uint64_t start, uint64_t length) const {
    auto v = m_view.slice(axis,
                          Symbolic{m_shape.graph(), Sym::Const((int64_t)start)},
                          Symbolic{m_shape.graph(), Sym::Const(1)});
    auto sh = m_shape;
    sh[axis] = Symbolic{m_shape.graph(), Sym::Const((int64_t)length)};
    return withView(std::move(sh), std::move(v));
  }

  HostTensor broadcastInDim(const TensorShape &to,
                            std::span<const int64_t> axesMap) const {
    auto v = m_view.broadcastInDim(m_shape.dims(), to.dims(), axesMap);
    return withView(to, std::move(v));
  }

  HostTensor clone() const { return materializeContiguous(); }

  bool sameStorageAs(const HostTensor &o) const {
    return m_store.get() == o.m_store.get();
  }

private:
  explicit HostTensor(TensorShape shape, TensorViewDesc view,
                      std::shared_ptr<HostTensorStorage> storage)
      : m_shape(shape), m_view(std::move(view)), m_store(std::move(storage)) {}
  TensorShape m_shape;
  TensorViewDesc m_view;
  std::shared_ptr<HostTensorStorage> m_store;
};

class DeviceTensor {
public:
  explicit DeviceTensor(std::size_t rank, vkcnn::Tensor handle)
      : m_rank(rank), m_handle(std::move(handle)) {
    assert(m_rank == 3 || m_rank == 4);
  }

  TensorShape shape() const {
    const auto &w = m_handle.width();
    const auto &h = m_handle.height();
    const auto &g = w.graph();
    assert(g.get() == h.graph().get());
    std::vector<Symbolic> dims;
    dims.reserve(m_rank);
    if (m_rank == 4) {
      // NCHW
      dims.push_back(Symbolic{g, Sym::Const(1)});
    }
    assert(m_handle.channels() > 0);
    dims.push_back(Symbolic{g, Sym::Const(m_handle.channels())});
    dims.push_back(h);
    dims.push_back(w);
    return TensorShape{g, std::move(dims)};
  }

  std::size_t rank() const { return m_rank; }

  const vkcnn::Tensor &handle() const { return m_handle; }
  vkcnn::Tensor &handle() { return m_handle; }

  const std::shared_ptr<SymGraph> &graph() const {
    assert(m_handle.width().graph().get() == m_handle.height().graph().get());
    return m_handle.width().graph();
  }

  bool sameHandleAs(const DeviceTensor &o) const {
    return m_handle.id() == o.m_handle.id();
  }

  std::optional<Dtype> type() const {
    auto ftype = m_handle.type();
    if (!ftype.has_value()) {
      return std::nullopt;
    }
    if (ftype == vkcnn::FloatType::F16) {
      return Dtype::Float16;
    } else if (ftype == vkcnn::FloatType::F32) {
      return Dtype::Float32;
    } else if (ftype == vkcnn::FloatType::F64) {
      return Dtype::Float64;
    }
    throw std::logic_error("unreachable");
  }

private:
  std::size_t m_rank; // i.e CHW or NCHW
  vkcnn::Tensor m_handle;
};

class Tensor {
  using Rep =
      std::variant<std::shared_ptr<HostTensor>, std::shared_ptr<DeviceTensor>>;

public:
  static Tensor Host(HostTensor hostTensor) {
    return Tensor{std::make_shared<HostTensor>(std::move(hostTensor))};
  }
  static Tensor Device(DeviceTensor deviceTensor) {
    return Tensor{std::make_shared<DeviceTensor>(std::move(deviceTensor))};
  }

  static Tensor Host(std::shared_ptr<HostTensor> hostTensor) {
    return Tensor{std::move(hostTensor)};
  }
  static Tensor Device(std::shared_ptr<DeviceTensor> deviceTensor) {
    return Tensor{std::move(deviceTensor)};
  }

  bool isDevice() const {
    return std::holds_alternative<std::shared_ptr<DeviceTensor>>(m_rep);
  }
  bool isHost() const {
    return std::holds_alternative<std::shared_ptr<HostTensor>>(m_rep);
  }

  const DeviceTensor &device() const {
    assert(isDevice());
    return *std::get<std::shared_ptr<DeviceTensor>>(m_rep);
  }
  const HostTensor &host() const {
    assert(isHost());
    return *std::get<std::shared_ptr<HostTensor>>(m_rep);
  }

  TensorShape shape() const {
    if (isDevice()) {
      return device().shape();
    } else if (isHost()) {
      return host().shape();
    } else {
      throw std::logic_error("unreachable");
    }
  }

  std::size_t rank() const {
    if (isDevice()) {
      return device().rank();
    } else if (isHost()) {
      return host().rank();
    } else {
      throw std::logic_error("unreachable");
    }
  }

  std::optional<Dtype> type() {
    if (isDevice()) {
      return device().type();
    } else if (isHost()) {
      return host().type();
    } else {
      throw std::logic_error("unreachable");
    }
  }

private:
  Tensor(Rep rep) : m_rep(std::move(rep)) {}
  Rep m_rep;
};

class Optional {
  // stub.
};

class Sequence {
  // stub.
};

class Map {
  // stub.
};

class SparseTensor {
  // stub.
};

class Opaque {
  // stub.
};

enum class ValueKind { Tensor, Optional, Sequence, Map, SparseTensor, Opaque };

class Value {
  using Rep =
      std::variant<Tensor, Optional, Sequence, Map, SparseTensor, Opaque>;

public:
  // ---- factories (no default-constructed "invalid" Value) ----
  static Value FromTensor(Tensor t) { return Value{Rep{std::move(t)}}; }
  static Value FromOptional(Optional v = {}) {
    return Value{Rep{std::move(v)}};
  }
  static Value FromSequence(Sequence v = {}) {
    return Value{Rep{std::move(v)}};
  }
  static Value FromMap(Map v = {}) { return Value{Rep{std::move(v)}}; }
  static Value FromSparseTensor(SparseTensor v = {}) {
    return Value{Rep{std::move(v)}};
  }
  static Value FromOpaque(Opaque v = {}) { return Value{Rep{std::move(v)}}; }

  // ---- kind / queries ----
  ValueKind kind() const {
    if (std::holds_alternative<Tensor>(m_rep))
      return ValueKind::Tensor;
    if (std::holds_alternative<Optional>(m_rep))
      return ValueKind::Optional;
    if (std::holds_alternative<Sequence>(m_rep))
      return ValueKind::Sequence;
    if (std::holds_alternative<Map>(m_rep))
      return ValueKind::Map;
    if (std::holds_alternative<SparseTensor>(m_rep))
      return ValueKind::SparseTensor;
    if (std::holds_alternative<Opaque>(m_rep))
      return ValueKind::Opaque;
    // unreachable
    assert(false);
    return ValueKind::Opaque;
  }

  bool isTensor() const { return std::holds_alternative<Tensor>(m_rep); }
  bool isOptional() const { return std::holds_alternative<Optional>(m_rep); }
  bool isSequence() const { return std::holds_alternative<Sequence>(m_rep); }
  bool isMap() const { return std::holds_alternative<Map>(m_rep); }
  bool isSparseTensor() const {
    return std::holds_alternative<SparseTensor>(m_rep);
  }
  bool isOpaque() const { return std::holds_alternative<Opaque>(m_rep); }

  // ---- accessors ----
  // Tensor is the only kind we actually support operationally.
  const Tensor &tensor() const {
    if (!isTensor())
      throw std::runtime_error("not supported: Value is not a Tensor");
    return std::get<Tensor>(m_rep);
  }
  Tensor &tensor() {
    if (!isTensor())
      throw std::runtime_error("not supported: Value is not a Tensor");
    return std::get<Tensor>(m_rep);
  }

  const Optional &optional() const { throwNS("Optional"); }
  const Sequence &sequence() const { throwNS("Sequence"); }
  const Map &map() const { throwNS("Map"); }
  const SparseTensor &sparse() const { throwNS("SparseTensor"); }
  const Opaque &opaque() const { throwNS("Opaque"); }

  static constexpr std::string_view kindName(ValueKind k) {
    switch (k) {
    case ValueKind::Tensor:
      return "Tensor";
    case ValueKind::Optional:
      return "Optional";
    case ValueKind::Sequence:
      return "Sequence";
    case ValueKind::Map:
      return "Map";
    case ValueKind::SparseTensor:
      return "SparseTensor";
    case ValueKind::Opaque:
      return "Opaque";
    }
    return "Unknown";
  }

private:
  explicit Value(Rep rep) : m_rep(std::move(rep)) {}

  [[noreturn]] static void throwNS(const char *what) {
    throw std::runtime_error(std::string("not supported: ") + what);
  }

  Rep m_rep;
};

struct GraphAttr { /* stub */
};

enum class AttributeKind {
  Int,
  Float,
  String,
  Tensor,
  Graph,
  Floats,
  Ints,
  Strings,
  Tensors,
  Graphs,
};

static constexpr std::string_view AttributeKind_name(AttributeKind k) noexcept {
  switch (k) {
  case AttributeKind::Int:
    return "Int";
  case AttributeKind::Float:
    return "Float";
  case AttributeKind::String:
    return "String";
  case AttributeKind::Tensor:
    return "Tensor";
  case AttributeKind::Graph:
    return "Graph";
  case AttributeKind::Floats:
    return "Floats";
  case AttributeKind::Ints:
    return "Ints";
  case AttributeKind::Strings:
    return "Strings";
  case AttributeKind::Tensors:
    return "Tensors";
  case AttributeKind::Graphs:
    return "Graphs";
  }
  return "Unknown";
}

class Attribute {
  using Rep = std::variant<std::int64_t,              // Int
                           float,                     // Float (ONNX float)
                           std::string,               // String
                           HostTensor,                // Tensor (host-only)
                           GraphAttr,                 // Graph (stub)
                           std::vector<float>,        // Floats
                           std::vector<std::int64_t>, // Ints
                           std::vector<std::string>,  // Strings
                           std::vector<HostTensor>,   // Tensors (host-only)
                           std::vector<GraphAttr>     // Graphs (stub)
                           >;

public:
  static Attribute Int(std::int64_t v) { return Attribute{Rep{v}}; }
  static Attribute Float(float v) { return Attribute{Rep{v}}; }
  static Attribute String(std::string v) {
    return Attribute{Rep{std::move(v)}};
  }
  static Attribute Tensor(const HostTensor &t) { return Attribute{Rep{t}}; }
  static Attribute Tensor(HostTensor &&t) {
    return Attribute{Rep{std::move(t)}};
  }
  static Attribute Graph(GraphAttr g = {}) {
    return Attribute{Rep{std::move(g)}};
  }

  static Attribute Ints(std::vector<std::int64_t> v) {
    return Attribute{Rep{std::move(v)}};
  }
  static Attribute Floats(std::vector<float> v) {
    return Attribute{Rep{std::move(v)}};
  }
  static Attribute Strings(std::vector<std::string> v) {
    return Attribute{Rep{std::move(v)}};
  }
  static Attribute Tensors(std::vector<HostTensor> v) {
    return Attribute{Rep{std::move(v)}};
  }
  static Attribute Graphs(std::vector<GraphAttr> v) {
    return Attribute{Rep{std::move(v)}};
  }

  // ---- kind ----
  AttributeKind kind() const {
    if (std::holds_alternative<std::int64_t>(m_rep))
      return AttributeKind::Int;
    if (std::holds_alternative<float>(m_rep))
      return AttributeKind::Float;
    if (std::holds_alternative<std::string>(m_rep))
      return AttributeKind::String;
    if (std::holds_alternative<HostTensor>(m_rep))
      return AttributeKind::Tensor;
    if (std::holds_alternative<GraphAttr>(m_rep))
      return AttributeKind::Graph;
    if (std::holds_alternative<std::vector<float>>(m_rep))
      return AttributeKind::Floats;
    if (std::holds_alternative<std::vector<std::int64_t>>(m_rep))
      return AttributeKind::Ints;
    if (std::holds_alternative<std::vector<std::string>>(m_rep))
      return AttributeKind::Strings;
    if (std::holds_alternative<std::vector<HostTensor>>(m_rep))
      return AttributeKind::Tensors;
    if (std::holds_alternative<std::vector<GraphAttr>>(m_rep))
      return AttributeKind::Graphs;
    assert(false);
    return AttributeKind::Ints;
  }

  // ---- predicates ----
  bool isInt() const { return std::holds_alternative<std::int64_t>(m_rep); }
  bool isFloat() const { return std::holds_alternative<float>(m_rep); }
  bool isString() const { return std::holds_alternative<std::string>(m_rep); }
  bool isTensor() const { return std::holds_alternative<HostTensor>(m_rep); }
  bool isGraph() const { return std::holds_alternative<GraphAttr>(m_rep); }
  bool isInts() const {
    return std::holds_alternative<std::vector<std::int64_t>>(m_rep);
  }
  bool isFloats() const {
    return std::holds_alternative<std::vector<float>>(m_rep);
  }
  bool isStrings() const {
    return std::holds_alternative<std::vector<std::string>>(m_rep);
  }
  bool isTensors() const {
    return std::holds_alternative<std::vector<HostTensor>>(m_rep);
  }
  bool isGraphs() const {
    return std::holds_alternative<std::vector<GraphAttr>>(m_rep);
  }

  // ---- accessors (throw if wrong kind) ----
  std::int64_t i() const {
    ensure(AttributeKind::Int);
    return std::get<std::int64_t>(m_rep);
  }
  float f() const {
    ensure(AttributeKind::Float);
    return std::get<float>(m_rep);
  }
  const std::string &s() const {
    ensure(AttributeKind::String);
    return std::get<std::string>(m_rep);
  }
  const HostTensor &t() const {
    ensure(AttributeKind::Tensor);
    return std::get<HostTensor>(m_rep);
  }
  const GraphAttr &g() const {
    ensure(AttributeKind::Graph);
    return std::get<GraphAttr>(m_rep);
  }
  const std::vector<std::int64_t> &ints() const {
    ensure(AttributeKind::Ints);
    return std::get<std::vector<std::int64_t>>(m_rep);
  }
  const std::vector<float> &floats() const {
    ensure(AttributeKind::Floats);
    return std::get<std::vector<float>>(m_rep);
  }
  const std::vector<std::string> &strings() const {
    ensure(AttributeKind::Strings);
    return std::get<std::vector<std::string>>(m_rep);
  }
  const std::vector<HostTensor> &tensors() const {
    ensure(AttributeKind::Tensors);
    return std::get<std::vector<HostTensor>>(m_rep);
  }
  const std::vector<GraphAttr> &graphs() const {
    ensure(AttributeKind::Graphs);
    return std::get<std::vector<GraphAttr>>(m_rep);
  }

  std::string_view kindName(AttributeKind k) noexcept {
    return AttributeKind_name(k);
  }

private:
  explicit Attribute(Rep rep) : m_rep(std::move(rep)) {}
  void ensure(AttributeKind k) const {
    if (kind() != k)
      throw std::runtime_error(
          fmt::format("Attribute: wrong kind. Expected {}, Got {}",
                      AttributeKind_name(k), AttributeKind_name(kind())));
  }
  Rep m_rep;
};

static void print_tensor(const Tensor &tensor) {
  if (tensor.isDevice()) {
    fmt::println("=====DeviceTensor======");
    auto device = tensor.device();
    auto shape = device.shape();
    fmt::print("Shape: {{");
    for (std::size_t i = 0; i < shape.dims().size(); ++i) {
      if (i != 0) {
        fmt::print(", ");
      }
      if (shape.dims()[i].isSymbolic()) {
        fmt::print("[{}]", shape.dims()[i].resolve().sym());
      } else {
        fmt::print("{}", shape.dims()[i].constant());
      }
    }
    fmt::println("}}");
  } else {
    fmt::println("=======HostTensor======");
    auto host = tensor.host();
    auto shape = host.shape();
    fmt::print("Shape: {{");
    for (std::size_t i = 0; i < shape.dims().size(); ++i) {
      if (i != 0) {
        fmt::print(", ");
      }
      if (shape.dims()[i].isSymbolic()) {
        fmt::print("[{}]", shape.dims()[i].resolve().sym());
      } else {
        fmt::print("{}", shape.dims()[i].constant());
      }
    }
    fmt::println("}}");

    auto view = host.view();
    fmt::print("Strides: {{");
    for (std::size_t i = 0; i < view.strides().size(); ++i) {
      if (i != 0) {
        fmt::print(", ");
      }
      if (view.strides()[i].isSymbolic()) {
        fmt::print("[{}]", view.strides()[i].resolve().sym());
      } else {
        fmt::print("{}", view.strides()[i].constant());
      }
    }
    fmt::println("}}");
    fmt::print("Offset = ");
    if (view.offset().isSymbolic()) {
      fmt::println("[{}]", view.offset().resolve().sym());
    } else {
      fmt::println("{}", view.offset().constant());
    }
    fmt::println("Dtype: {}", dtype_to_string(host.type()));
    auto store = host.storage();
    switch (host.type()) {
    case Dtype::Undefined:
    case Dtype::Int8: {
      for (std::size_t i = 0; i < store->i8().size(); ++i) {
        fmt::println("[{}] : {}", i, store->i8()[i]);
      }
      break;
    }
    case Dtype::Int16: {
      for (std::size_t i = 0; i < store->i16().size(); ++i) {
        fmt::println("[{}] : {}", i, store->i16()[i]);
      }
      break;
    }
    case Dtype::Int32: {
      for (std::size_t i = 0; i < store->i32().size(); ++i) {
        fmt::println("[{}] : {}", i, store->i32()[i]);
      }
      break;
    }
    case Dtype::Int64: {
      for (std::size_t i = 0; i < store->i64().size(); ++i) {
        fmt::println("[{}] : {}", i, store->i64()[i]);
      }
      break;
    }
    case Dtype::Uint8: {
      for (std::size_t i = 0; i < store->u8().size(); ++i) {
        fmt::println("[{}] : {}", i, store->u8()[i]);
      }
      break;
    }
    case Dtype::Uint16: {
      for (std::size_t i = 0; i < store->u16().size(); ++i) {
        fmt::println("[{}] : {}", i, store->u16()[i]);
      }
      break;
    }
    case Dtype::Uint32: {
      for (std::size_t i = 0; i < store->u32().size(); ++i) {
        fmt::println("[{}] : {}", i, store->u32()[i]);
      }
      break;
    }
    case Dtype::Uint64: {
      for (std::size_t i = 0; i < store->u64().size(); ++i) {
        fmt::println("[{}] : {}", i, store->u64()[i]);
      }
      break;
    }
    case Dtype::Float64: {
      for (std::size_t i = 0; i < store->f64().size(); ++i) {
        fmt::println("[{}] : {}", i, store->f64()[i]);
      }
      break;
    }
    case Dtype::Float32: {
      for (std::size_t i = 0; i < store->f32().size(); ++i) {
        fmt::println("[{}] : {}", i, store->f32()[i]);
      }
      break;
    }
    case Dtype::Float16:
    case Dtype::String:
    case Dtype::Bool:
    case Dtype::Sym: {
      for (std::size_t i = 0; i < store->sym().size(); ++i) {
        if (store->sym()[i].isSymbolic()) {
          fmt::println("[{}] : [{}]", i, store->sym()[i].sym());
        } else {
          fmt::println("[{}] : {}", i, store->sym()[i].constant());
        }
      }
      // shape.graph()->debugDump();
      break;
    }
    }
  }
}

} // namespace vkcnn::details
