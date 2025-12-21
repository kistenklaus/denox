#include "denox/compiler/frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor>
pad(ImportState &state, memory::span<const memory::optional<Tensor>> inputs,
    std::size_t outputCount,
    const memory::hash_map<memory::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() < 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\" expects at least 2 inputs (data, pads).", nodeName));
  if (inputs.size() > 4)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\": too many inputs (max 4).", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &dataT = *inputs[0];

  const Tensor &padsT = *inputs[1];
  const bool hasConstVal = (inputs.size() >= 3) && inputs[2].has_value();
  const bool hasAxes = (inputs.size() >= 4) && inputs[3].has_value();

  // ---- mode attribute (string) ----
  memory::string modeStr = "constant";
  if (auto it = attributes.find("mode"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": attribute 'mode' must be string.", nodeName));
    modeStr = it->second.s();
  }
  enum class HostPadMode { Constant, Edge, Reflect };
  HostPadMode hostMode;
  PaddingMode devMode;
  if (modeStr == "constant") {
    hostMode = HostPadMode::Constant;
    devMode = PaddingMode::Zero;
  } else if (modeStr == "edge") {
    hostMode = HostPadMode::Edge;
    devMode = PaddingMode::Edge;
  } else if (modeStr == "reflect") {
    hostMode = HostPadMode::Reflect; /* device unsupported */
  } else {
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": unsupported mode '{}'.", nodeName, modeStr));
  }

  // ---- pads must be HostTensor, 1-D ----
  if (!padsT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": 'pads' must be a host tensor.", nodeName));
  const HostTensor &padsHT = padsT.host();
  const auto padsShapeU64 = padsHT.shape().toU64();
  if (padsShapeU64.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\": 'pads' must be 1-D.", nodeName));

  // ---- axes (optional), must be Host INT64 1-D if present ----
  memory::vector<std::int64_t> axes;
  if (hasAxes) {
    const Tensor &axesT = *inputs[3];
    if (!axesT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'axes' must be a host tensor.", nodeName));
    const HostTensor &axesHT = axesT.host();
    if (!axesHT.isConstant() || axesHT.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'axes' must be constant INT64.", nodeName));
    const auto axesSpan = axesHT.storage()->i64();
    axes.assign(axesSpan.begin(), axesSpan.end());
  }

  // ---- device path ----
  if (dataT.isDevice()) {
    if (hostMode == HostPadMode::Reflect)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'reflect' mode not supported for device tensors.",
          nodeName));

    const DeviceTensor &devIn = dataT.device();
    const TensorShape inS = devIn.shape();
    const std::size_t r = inS.rank();
    if (r != 3 && r != 4)
      throw std::runtime_error(
          fmt::format("vkcnn: Pad \"{}\": device tensor must be rank 3 (CHW) "
                      "or 4 (NCHW); got {}.",
                      nodeName, r));

    // Parse pads -> per-axis before/after (as Sym). If 'axes' present, pads
    // length must be 2*axes.size().
    const std::size_t padLen = padsHT.sizeElemsIfStatic();
    if (padsHT.type() == Dtype::Int64) {
      if (!padsHT.isConstant())
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": device path requires constant 'pads' if INT64.",
            nodeName));
    } else if (padsHT.type() != Dtype::Sym) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'pads' must be INT64 or SYM.", nodeName));
    }

    std::size_t expect = hasAxes ? (2 * axes.size()) : (2 * r);
    if (padLen != expect)
      throw std::runtime_error(
          fmt::format("vkcnn: Pad \"{}\": 'pads' length must be {} (got {}).",
                      nodeName, expect, padLen));

    // Build full-length vectors (rank) of Sym zeros, then fill from axes/pads.
    memory::vector<Sym> before(r, Sym::Const(0));
    memory::vector<Sym> after(r, Sym::Const(0));

    auto put_pad = [&](std::size_t ax, const Sym &b,
                       const Sym &a) {
      before[ax] = b;
      after[ax] = a;
    };

    if (hasAxes) {
      // normalize axes and fill in order
      memory::vector<std::size_t> nAxes;
      nAxes.reserve(axes.size());
      for (auto v : axes) {
        if (v < 0)
          v += static_cast<std::int64_t>(r);
        if (v < 0 || v >= static_cast<std::int64_t>(r))
          throw std::runtime_error(fmt::format(
              "vkcnn: Pad \"{}\": axis {} out of range for rank {}.", nodeName,
              v, r));
        nAxes.push_back(static_cast<std::size_t>(v));
      }
      // Gather pads values
      if (padsHT.type() == Dtype::Int64) {
        auto p = padsHT.storage()->i64();
        for (size_t i = 0; i < nAxes.size(); ++i) {
          std::int64_t b = p[i];
          std::int64_t a = p[i + nAxes.size()];
          if (b < 0 || a < 0)
            throw std::runtime_error(fmt::format(
                "vkcnn: Pad \"{}\": negative pads not supported on device.",
                nodeName));
          put_pad(nAxes[i], Sym::Const(b), Sym::Const(a));
        }
      } else { // SYM
        auto p = padsHT.storage()->sym();
        for (size_t i = 0; i < nAxes.size(); ++i) {
          put_pad(nAxes[i], p[i], p[i + nAxes.size()]);
        }
      }
    } else {
      // pads for all axes (order: all befores, then all afters)
      if (padsHT.type() == Dtype::Int64) {
        auto p = padsHT.storage()->i64();
        for (size_t ax = 0; ax < r; ++ax) {
          std::int64_t b = p[ax];
          std::int64_t a = p[ax + r];
          if (b < 0 || a < 0)
            throw std::runtime_error(fmt::format(
                "vkcnn: Pad \"{}\": negative pads not supported on device.",
                nodeName));
          before[ax] = Sym::Const(b);
          after[ax] = Sym::Const(a);
        }
      } else {
        auto p = padsHT.storage()->sym();
        for (size_t ax = 0; ax < r; ++ax) {
          before[ax] = p[ax];
          after[ax] = p[ax + r];
        }
      }
    }

    // Only H/W padding supported; others must be zero.
    const std::size_t axH = (r == 4) ? 2u : 1u;
    const std::size_t axW = (r == 4) ? 3u : 2u;
    const Sym ZERO = Sym::Const(0);

    auto isKnownZero = [&](const Sym &s) -> bool {
      // Accept literal constant zero or symbolic equality to ZERO.
      return (s.isConstant() && s.constant() == 0) || (s == ZERO);
    };

    for (size_t ax = 0; ax < r; ++ax) {
      if (ax == axH || ax == axW)
        continue;
      if (!isKnownZero(before[ax]) || !isKnownZero(after[ax])) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": device path supports padding only H/W axes.",
            nodeName));
      }
    }

    // constant mode: only zero value allowed
    if (hostMode == HostPadMode::Constant && hasConstVal) {
      const Tensor &cvT = *inputs[2];
      if (!cvT.isHost())
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": 'constant_value' must be a host tensor.",
            nodeName));
      const HostTensor &cv = cvT.host();
      if (!cv.isConstant() || cv.sizeElemsIfStatic() != 1)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": 'constant_value' must be scalar & constant.",
            nodeName));
      // accept int/float == 0 only
      bool ok = false;
      switch (cv.type().kind()) {
      case DtypeKind::Int64:
        ok = (cv.storage()->i64()[0] == 0);
        break;
      case DtypeKind::Int32:
        ok = (cv.storage()->i32()[0] == 0);
        break;
      case DtypeKind::Float32:
        ok = (cv.storage()->f32()[0] == 0.0f);
        break;
      case DtypeKind::Float64:
        ok = (cv.storage()->f64()[0] == 0.0);
        break;
      case DtypeKind::Uint64:
        ok = (cv.storage()->u64()[0] == 0);
        break;
      case DtypeKind::Uint32:
        ok = (cv.storage()->u32()[0] == 0);
        break;
      case DtypeKind::Uint16:
        ok = (cv.storage()->u16()[0] == 0);
        break;
      case DtypeKind::Uint8:
        ok = (cv.storage()->u8()[0] == 0);
        break;
      case DtypeKind::Int16:
        ok = (cv.storage()->i16()[0] == 0);
        break;
      case DtypeKind::Int8:
        ok = (cv.storage()->i8()[0] == 0);
        break;
      case DtypeKind::Bool:
        ok = (cv.storage()->boolean()[0] == 0);
        break;
      case DtypeKind::Float16: {
        // treat any nonzero 16-bit as nonzero; simplest check: bitwise compare
        // to 0
        const memory::f16 v = cv.storage()->f16()[0];
        ok = (std::memcmp(&v, "\0\0", 2) == 0);
        break;
      }
      case DtypeKind::String:
      case DtypeKind::Sym:
      case DtypeKind::Undefined:
      default:
        ok = false;
        break;
      }
      if (!ok)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": device constant mode supports only value==0.",
            nodeName));
    }

    // Call backend
    const Sym T = before[axH], B = after[axH], L = before[axW],
                        R = after[axW];

    compiler::TensorHandle outH =
        state.output.pad(devIn.handle(), L, R, T, B, devMode);
    DeviceTensor outDev(r, std::move(outH));

    return {Tensor::Device(std::move(outDev))};
  }

  // ---- host path ----
  const HostTensor &hin = dataT.host();
  if (!hin.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": dynamic host tensor unsupported.", nodeName));

  const auto inU64 = hin.shape().toU64();
  const std::size_t r = inU64.size();

  // Build before/after (size r) from pads (+axes). Require constant INT64 pads
  // to materialize.
  if (!padsHT.isConstant() || padsHT.type() != Dtype::Int64)
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": host path requires constant INT64 'pads'.",
        nodeName));

  const auto p = padsHT.storage()->i64();
  std::size_t expect = hasAxes ? (2 * axes.size()) : (2 * r);
  if (p.size() != expect)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\": 'pads' length must be {} (got {}).",
                    nodeName, expect, p.size()));

  memory::vector<std::uint64_t> before(r, 0), after(r, 0);
  if (hasAxes) {
    // normalize axes
    memory::vector<size_t> nAxes;
    nAxes.reserve(axes.size());
    for (auto v : axes) {
      if (v < 0)
        v += static_cast<std::int64_t>(r);
      if (v < 0 || v >= static_cast<std::int64_t>(r))
        throw std::runtime_error(
            fmt::format("vkcnn: Pad \"{}\": axis {} out of range for rank {}.",
                        nodeName, v, r));
      nAxes.push_back(static_cast<size_t>(v));
    }
    for (size_t i = 0; i < nAxes.size(); ++i) {
      if (p[i] < 0 || p[i + nAxes.size()] < 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": negative pads not supported on host.",
            nodeName));
      before[nAxes[i]] = static_cast<std::uint64_t>(p[i]);
      after[nAxes[i]] = static_cast<std::uint64_t>(p[i + nAxes.size()]);
    }
  } else {
    for (size_t ax = 0; ax < r; ++ax) {
      if (p[ax] < 0 || p[ax + r] < 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": negative pads not supported on host.",
            nodeName));
      before[ax] = static_cast<std::uint64_t>(p[ax]);
      after[ax] = static_cast<std::uint64_t>(p[ax + r]);
    }
  }

  // Output dims
  memory::vector<std::uint64_t> outDims(r);
  for (std::size_t ax = 0; ax < r; ++ax)
    outDims[ax] = inU64[ax] + before[ax] + after[ax];

  // Build out shape
  auto g = hin.shape().graph();
  memory::vector<compiler::Symbolic> outSyms(r);
  for (size_t ax = 0; ax < r; ++ax)
    outSyms[ax] = compiler::Symbolic{
        g, Sym::Const(static_cast<std::int64_t>(outDims[ax]))};
  TensorShape outShape{g, std::move(outSyms)};

  // Dtype & alloc
  const Dtype dt = hin.type();
  const std::size_t esz = hin.elemSize();
  std::size_t total = 1;
  for (auto d : outDims)
    total *= static_cast<std::size_t>(d);

  std::shared_ptr<HostTensorStorage> outStore;
  if (dt == Dtype::String) {
    auto table = static_cast<char **>(std::malloc(total * sizeof(char *)));
    if (!table)
      throw std::bad_alloc();
    outStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::String, table, total * sizeof(char *)));
  } else {
    void *raw = std::malloc(total * esz);
    if (!raw)
      throw std::bad_alloc();
    outStore = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, total * esz));
  }

  // constant_value (host-mode only)
  long double constScalar = 0.0L;
  memory::string constString; // for String dtype
  if (hostMode == HostPadMode::Constant && hasConstVal) {
    const Tensor &cvT = *inputs[2];
    if (!cvT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'constant_value' must be a host tensor.",
          nodeName));
    const HostTensor &cv = cvT.host();
    if (!cv.isConstant() || cv.sizeElemsIfStatic() != 1)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'constant_value' must be scalar & constant.",
          nodeName));
    if (dt == Dtype::String) {
      if (cv.type() != Dtype::String)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": constant_value type must match (string).",
            nodeName));
      const char *s = cv.storage()->strs()[0];
      constString.assign(s ? s : "");
    } else {
      // accept any numeric
      switch (cv.type().kind()) {
      case DtypeKind::Float32:
        constScalar = static_cast<long double>(cv.storage()->f32()[0]);
        break;
      case DtypeKind::Float64:
        constScalar = static_cast<long double>(cv.storage()->f64()[0]);
        break;
      case DtypeKind::Float16: {
        float f = static_cast<float>(
            cv.storage()->f16()[0]); // relies on f16 -> float operator/ctor
        constScalar = static_cast<long double>(f);
        break;
      }
      case DtypeKind::Int64:
        constScalar = static_cast<long double>(cv.storage()->i64()[0]);
        break;
      case DtypeKind::Int32:
        constScalar = static_cast<long double>(cv.storage()->i32()[0]);
        break;
      case DtypeKind::Int16:
        constScalar = static_cast<long double>(cv.storage()->i16()[0]);
        break;
      case DtypeKind::Int8:
        constScalar = static_cast<long double>(cv.storage()->i8()[0]);
        break;
      case DtypeKind::Uint64:
        constScalar = static_cast<long double>(cv.storage()->u64()[0]);
        break;
      case DtypeKind::Uint32:
        constScalar = static_cast<long double>(cv.storage()->u32()[0]);
        break;
      case DtypeKind::Uint16:
        constScalar = static_cast<long double>(cv.storage()->u16()[0]);
        break;
      case DtypeKind::Uint8:
        constScalar = static_cast<long double>(cv.storage()->u8()[0]);
        break;
      case DtypeKind::Bool:
        constScalar = (cv.storage()->boolean()[0] ? 1.0L : 0.0L);
        break;
      case DtypeKind::String:
      case DtypeKind::Sym:
      case DtypeKind::Undefined:
      default:
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": unsupported constant_value dtype.", nodeName));
      }
    }
  }

  // helpers for host copy
  auto clamp = [](long long v, long long lo, long long hi) -> long long {
    return v < lo ? lo : (v > hi ? hi : v);
  };
  auto reflect_map = [](long long p, long long n) -> long long {
    if (n <= 1)
      return 0;
    long long m = 2 * (n - 1);
    long long q = p % m;
    if (q < 0)
      q += m;
    if (q >= n)
      q = m - q;
    return q;
  };

  const auto *srcBase = static_cast<const std::byte *>(hin.storage()->data());
  std::byte *dstBytes = (dt == Dtype::String)
                            ? nullptr
                            : static_cast<std::byte *>(outStore->data());
  char **dstStrs =
      (dt == Dtype::String) ? static_cast<char **>(outStore->data()) : nullptr;

  // write one constant element (for numeric)
  auto write_const_one = [&](std::byte *&out) {
    switch (dt.kind()) {
    case DtypeKind::Bool: {
      std::uint8_t v = (constScalar != 0.0L) ? 1u : 0u;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Int8: {
      std::int8_t v = static_cast<std::int8_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Int16: {
      std::int16_t v = static_cast<std::int16_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Int32: {
      std::int32_t v = static_cast<std::int32_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Int64: {
      std::int64_t v = static_cast<std::int64_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Uint8: {
      std::uint8_t v = static_cast<std::uint8_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Uint16: {
      std::uint16_t v = static_cast<std::uint16_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Uint32: {
      std::uint32_t v = static_cast<std::uint32_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Uint64: {
      std::uint64_t v = static_cast<std::uint64_t>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Float16: {
      memory::f16 v = memory::f16{static_cast<float>(constScalar)};
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Float32: {
      float v = static_cast<float>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::Float64: {
      double v = static_cast<double>(constScalar);
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case DtypeKind::String:
    case DtypeKind::Sym:
    case DtypeKind::Undefined:
    default:
      throw std::runtime_error(
          "vkcnn: internal: write_const_one unsupported dtype.");
    }
  };

  // iterate output
  memory::vector<std::uint64_t> oIdx(r, 0), sIdx(r, 0);
  std::size_t written = 0;

  auto inc = [&](memory::vector<std::uint64_t> &idx,
                 memory::span<const std::uint64_t> dims) -> bool {
    if (idx.empty())
      return false;
    std::size_t ax = idx.size();
    while (ax > 0) {
      --ax;
      if (++idx[ax] < dims[ax])
        return true;
      idx[ax] = 0;
    }
    return false;
  };

  while (true) {
    bool outIsConst = false;
    for (size_t ax = 0; ax < r; ++ax) {
      const long long n = static_cast<long long>(inU64[ax]);
      const long long bef = static_cast<long long>(before[ax]);
      const long long pos = static_cast<long long>(oIdx[ax]);
      const long long rel = pos - bef;
      if (hostMode == HostPadMode::Constant) {
        if (rel < 0 || rel >= n) {
          outIsConst = true;
          break;
        }
        sIdx[ax] = static_cast<std::uint64_t>(rel);
      } else if (hostMode == HostPadMode::Edge) {
        sIdx[ax] = static_cast<std::uint64_t>(clamp(rel, 0, n - 1));
      } else { // Reflect
        sIdx[ax] = static_cast<std::uint64_t>(reflect_map(rel, n));
      }
    }

    if (dt == Dtype::String) {
      if (hostMode == HostPadMode::Constant && outIsConst) {
        const memory::string &s = hasConstVal ? constString : memory::string();
        char *cp = static_cast<char *>(std::malloc(s.size() + 1));
        if (!cp)
          throw std::bad_alloc();
        std::memcpy(cp, s.data(), s.size());
        cp[s.size()] = '\0';
        dstStrs[written++] = cp;
      } else {
        const std::size_t srcElem = hin.view().constIndexOf(sIdx);
        const char *src =
            static_cast<char *const *>(hin.storage()->data())[srcElem];
        const std::size_t len = std::strlen(src);
        char *cp = static_cast<char *>(std::malloc(len + 1));
        if (!cp)
          throw std::bad_alloc();
        std::memcpy(cp, src, len + 1);
        dstStrs[written++] = cp;
      }
    } else {
      if (hostMode == HostPadMode::Constant && outIsConst) {
        write_const_one(dstBytes);
        ++written;
      } else {
        const std::size_t srcElem = hin.view().constIndexOf(sIdx);
        const std::size_t srcByte = srcElem * esz;
        std::memcpy(dstBytes, srcBase + srcByte, esz);
        dstBytes += esz;
        ++written;
      }
    }

    if (!inc(oIdx, outDims))
      break;
  }

  HostTensor outHT(outShape, std::move(outStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace denox::onnx::details::ops
