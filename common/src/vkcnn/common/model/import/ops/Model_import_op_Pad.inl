#pragma once

#include "vkcnn/common/model/import/Model_import_dtype.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/base.h>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Pad(ImportState &state, std::span<const std::optional<Tensor>> inputs,
              std::size_t outputCount,
              const std::unordered_map<std::string, Attribute> &attributes,
              [[maybe_unused]] opset_version version,
              const onnx::NodeProto &node) {

  // debug path:
  {
    std::size_t rank = inputs[0]->rank();
    fmt::println("input-rank: {}", rank);

    fmt::println("Pads:");

    fmt::println("Shape:");
    auto dims = inputs[1]->host().shape().dims();
    for (std::size_t s = 0; s < dims.size(); ++s) {
      if (dims[s].isSymbolic()) {
        fmt::println("[{}]: Sym", s);
      } else {
        fmt::println("[{}]: {}", s, dims[s].constant());
      }
    }

    fmt::println("Strides:");
    auto stride = inputs[1]->host().view().strides();
    for (std::size_t s = 0; s < stride.size(); ++s) {
      if (stride[s].isSymbolic()) {
        fmt::println("[{}]: Sym", s);
      } else {
        fmt::println("[{}]: {}", s, stride[s].constant());
      }
    }
    auto of = inputs[1]->host().view().offset();
    if (of.isSymbolic()) {
      fmt::println("offset = Sym");
    } else {
      fmt::println("offset = {}", of.constant());
    }
      
    fmt::println("Storage:");
    auto pads = inputs[1]->host().storage()->sym();
    for (std::size_t i = 0; i < pads.size(); ++i) {
      if (pads[i].isSymbolic()) {
        fmt::println("[{}]: Sym", i);
      } else {
        fmt::println("[{}]: {}", i, pads[i].constant());
      }
    }
  }

  // ---- arity ----
  if (inputs.size() < 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\" expects at least 2 inputs (data, pads).",
                    node.name()));
  if (inputs.size() > 4)
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": too many inputs (max 4).", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];
  const Tensor &padsT = *inputs[1];
  const bool hasConstVal = (inputs.size() >= 3) && inputs[2].has_value();
  const bool hasAxes = (inputs.size() >= 4) && inputs[3].has_value();

  // ---- mode attribute (string) ----
  std::string modeStr = "constant";
  if (auto it = attributes.find("mode"); it != attributes.end()) {
    if (!it->second.isString())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": attribute 'mode' must be string.", node.name()));
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
        "vkcnn: Pad \"{}\": unsupported mode '{}'.", node.name(), modeStr));
  }

  // ---- pads must be HostTensor, 1-D ----
  if (!padsT.isHost())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": 'pads' must be a host tensor.", node.name()));
  const HostTensor &padsHT = padsT.host();
  const auto padsShapeU64 = padsHT.shape().toU64();
  if (padsShapeU64.size() != 1)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\": 'pads' must be 1-D.", node.name()));

  // ---- axes (optional), must be Host INT64 1-D if present ----
  std::vector<std::int64_t> axes;
  if (hasAxes) {
    const Tensor &axesT = *inputs[3];
    if (!axesT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'axes' must be a host tensor.", node.name()));
    const HostTensor &axesHT = axesT.host();
    if (!axesHT.isConstant() || axesHT.type() != Dtype::Int64)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'axes' must be constant INT64.", node.name()));
    const auto axesSpan = axesHT.storage()->i64();
    axes.assign(axesSpan.begin(), axesSpan.end());
  }

  // ---- device path ----
  if (dataT.isDevice()) {
    if (hostMode == HostPadMode::Reflect)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'reflect' mode not supported for device tensors.",
          node.name()));

    const DeviceTensor &devIn = dataT.device();
    const TensorShape inS = devIn.shape();
    const size_t r = inS.rank();
    if (r != 3 && r != 4)
      throw std::runtime_error(
          fmt::format("vkcnn: Pad \"{}\": device tensor must be rank 3 (CHW) "
                      "or 4 (NCHW); got {}.",
                      node.name(), r));

    // Parse pads -> per-axis before/after (as Sym). If 'axes' present, pads
    // length must be 2*axes.size().
    const size_t padLen = padsHT.sizeElemsIfStatic();
    if (padsHT.type() == Dtype::Int64) {
      if (!padsHT.isConstant())
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": device path requires constant 'pads' if INT64.",
            node.name()));
    } else if (padsHT.type() != Dtype::Sym) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'pads' must be INT64 or SYM.", node.name()));
    }

    size_t expect = hasAxes ? (2 * axes.size()) : (2 * r);
    if (padLen != expect)
      throw std::runtime_error(
          fmt::format("vkcnn: Pad \"{}\": 'pads' length must be {} (got {}).",
                      node.name(), expect, padLen));

    // Build full-length vectors (rank) of Sym zeros, then fill from axes/pads.
    std::vector<Sym> before(r, Sym::Const(0));
    std::vector<Sym> after(r, Sym::Const(0));

    auto put_pad = [&](size_t ax, const Sym &b, const Sym &a) {
      before[ax] = b;
      after[ax] = a;
    };

    if (hasAxes) {
      // normalize axes and fill in order
      std::vector<size_t> nAxes;
      nAxes.reserve(axes.size());
      for (auto v : axes) {
        if (v < 0)
          v += static_cast<std::int64_t>(r);
        if (v < 0 || v >= static_cast<std::int64_t>(r))
          throw std::runtime_error(fmt::format(
              "vkcnn: Pad \"{}\": axis {} out of range for rank {}.",
              node.name(), v, r));
        nAxes.push_back(static_cast<size_t>(v));
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
                node.name()));
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
                node.name()));
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
    const size_t axH = (r == 4) ? 2u : 1u;
    const size_t axW = (r == 4) ? 3u : 2u;
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
            node.name()));
      }
    }

    // constant mode: only zero value allowed
    if (hostMode == HostPadMode::Constant && hasConstVal) {
      const Tensor &cvT = *inputs[2];
      if (!cvT.isHost())
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": 'constant_value' must be a host tensor.",
            node.name()));
      const HostTensor &cv = cvT.host();
      if (!cv.isConstant() || cv.sizeElemsIfStatic() != 1)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": 'constant_value' must be scalar & constant.",
            node.name()));
      // accept int/float == 0 only
      bool ok = false;
      switch (cv.type()) {
      case Dtype::Int64:
        ok = (cv.storage()->i64()[0] == 0);
        break;
      case Dtype::Int32:
        ok = (cv.storage()->i32()[0] == 0);
        break;
      case Dtype::Float32:
        ok = (cv.storage()->f32()[0] == 0.0f);
        break;
      case Dtype::Float64:
        ok = (cv.storage()->f64()[0] == 0.0);
        break;
      case Dtype::Uint64:
        ok = (cv.storage()->u64()[0] == 0);
        break;
      case Dtype::Uint32:
        ok = (cv.storage()->u32()[0] == 0);
        break;
      case Dtype::Uint16:
        ok = (cv.storage()->u16()[0] == 0);
        break;
      case Dtype::Uint8:
        ok = (cv.storage()->u8()[0] == 0);
        break;
      case Dtype::Int16:
        ok = (cv.storage()->i16()[0] == 0);
        break;
      case Dtype::Int8:
        ok = (cv.storage()->i8()[0] == 0);
        break;
      case Dtype::Bool:
        ok = (cv.storage()->boolean()[0] == 0);
        break;
      case Dtype::Float16: {
        // treat any nonzero 16-bit as nonzero; simplest check: bitwise compare
        // to 0
        const vkcnn::f16 v = cv.storage()->f16()[0];
        ok = (std::memcmp(&v, "\0\0", 2) == 0);
        break;
      }
      default:
        ok = false;
        break;
      }
      if (!ok)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": device constant mode supports only value==0.",
            node.name()));
    }

    // Call backend
    const Sym T = before[axH], B = after[axH], L = before[axW], R = after[axW];
    vkcnn::Tensor outH = state.output.pad(devIn.handle(), L, R, T, B, devMode);
    DeviceTensor outDev(r, std::move(outH));
    return {Tensor::Device(std::move(outDev))};
  }

  // ---- host path ----
  const HostTensor &hin = dataT.host();
  if (!hin.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": dynamic host tensor unsupported.", node.name()));

  const auto inU64 = hin.shape().toU64();
  const size_t r = inU64.size();

  // Build before/after (size r) from pads (+axes). Require constant INT64 pads
  // to materialize.
  if (!padsHT.isConstant() || padsHT.type() != Dtype::Int64)
    throw std::runtime_error(fmt::format(
        "vkcnn: Pad \"{}\": host path requires constant INT64 'pads'.",
        node.name()));

  const auto p = padsHT.storage()->i64();
  size_t expect = hasAxes ? (2 * axes.size()) : (2 * r);
  if (p.size() != expect)
    throw std::runtime_error(
        fmt::format("vkcnn: Pad \"{}\": 'pads' length must be {} (got {}).",
                    node.name(), expect, p.size()));

  std::vector<std::uint64_t> before(r, 0), after(r, 0);
  if (hasAxes) {
    // normalize axes
    std::vector<size_t> nAxes;
    nAxes.reserve(axes.size());
    for (auto v : axes) {
      if (v < 0)
        v += static_cast<std::int64_t>(r);
      if (v < 0 || v >= static_cast<std::int64_t>(r))
        throw std::runtime_error(
            fmt::format("vkcnn: Pad \"{}\": axis {} out of range for rank {}.",
                        node.name(), v, r));
      nAxes.push_back(static_cast<size_t>(v));
    }
    for (size_t i = 0; i < nAxes.size(); ++i) {
      if (p[i] < 0 || p[i + nAxes.size()] < 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": negative pads not supported on host.",
            node.name()));
      before[nAxes[i]] = static_cast<std::uint64_t>(p[i]);
      after[nAxes[i]] = static_cast<std::uint64_t>(p[i + nAxes.size()]);
    }
  } else {
    for (size_t ax = 0; ax < r; ++ax) {
      if (p[ax] < 0 || p[ax + r] < 0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": negative pads not supported on host.",
            node.name()));
      before[ax] = static_cast<std::uint64_t>(p[ax]);
      after[ax] = static_cast<std::uint64_t>(p[ax + r]);
    }
  }

  // Output dims
  std::vector<std::uint64_t> outDims(r);
  for (size_t ax = 0; ax < r; ++ax)
    outDims[ax] = inU64[ax] + before[ax] + after[ax];

  // Build out shape
  auto g = hin.shape().graph();
  std::vector<Symbolic> outSyms(r);
  for (size_t ax = 0; ax < r; ++ax)
    outSyms[ax] =
        Symbolic{g, Sym::Const(static_cast<std::int64_t>(outDims[ax]))};
  TensorShape outShape{g, std::move(outSyms)};

  // Dtype & alloc
  const Dtype dt = hin.type();
  const size_t esz = hin.elemSize();
  std::size_t total = 1;
  for (auto d : outDims)
    total *= (size_t)d;

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
  std::string constString; // for String dtype
  if (hostMode == HostPadMode::Constant && hasConstVal) {
    const Tensor &cvT = *inputs[2];
    if (!cvT.isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'constant_value' must be a host tensor.",
          node.name()));
    const HostTensor &cv = cvT.host();
    if (!cv.isConstant() || cv.sizeElemsIfStatic() != 1)
      throw std::runtime_error(fmt::format(
          "vkcnn: Pad \"{}\": 'constant_value' must be scalar & constant.",
          node.name()));
    if (dt == Dtype::String) {
      if (cv.type() != Dtype::String)
        throw std::runtime_error(fmt::format(
            "vkcnn: Pad \"{}\": constant_value type must match (string).",
            node.name()));
      const char *s = cv.storage()->strs()[0];
      constString.assign(s ? s : "");
    } else {
      // accept any numeric
      switch (cv.type()) {
      case Dtype::Float32:
        constScalar = (long double)cv.storage()->f32()[0];
        break;
      case Dtype::Float64:
        constScalar = (long double)cv.storage()->f64()[0];
        break;
      case Dtype::Float16: {
        float f = (float)cv.storage()
                      ->f16()[0]; // relies on f16 -> float operator/ctor
        constScalar = (long double)f;
        break;
      }
      case Dtype::Int64:
        constScalar = (long double)cv.storage()->i64()[0];
        break;
      case Dtype::Int32:
        constScalar = (long double)cv.storage()->i32()[0];
        break;
      case Dtype::Int16:
        constScalar = (long double)cv.storage()->i16()[0];
        break;
      case Dtype::Int8:
        constScalar = (long double)cv.storage()->i8()[0];
        break;
      case Dtype::Uint64:
        constScalar = (long double)cv.storage()->u64()[0];
        break;
      case Dtype::Uint32:
        constScalar = (long double)cv.storage()->u32()[0];
        break;
      case Dtype::Uint16:
        constScalar = (long double)cv.storage()->u16()[0];
        break;
      case Dtype::Uint8:
        constScalar = (long double)cv.storage()->u8()[0];
        break;
      case Dtype::Bool:
        constScalar = (cv.storage()->boolean()[0] ? 1.0L : 0.0L);
        break;
      default:
        throw std::runtime_error(
            fmt::format("vkcnn: Pad \"{}\": unsupported constant_value dtype.",
                        node.name()));
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
    switch (dt) {
    case Dtype::Bool: {
      std::uint8_t v = (constScalar != 0.0L) ? 1u : 0u;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Int8: {
      std::int8_t v = (std::int8_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Int16: {
      std::int16_t v = (std::int16_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Int32: {
      std::int32_t v = (std::int32_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Int64: {
      std::int64_t v = (std::int64_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Uint8: {
      std::uint8_t v = (std::uint8_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Uint16: {
      std::uint16_t v = (std::uint16_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Uint32: {
      std::uint32_t v = (std::uint32_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Uint64: {
      std::uint64_t v = (std::uint64_t)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Float16: {
      vkcnn::f16 v = vkcnn::f16{static_cast<float>(constScalar)};
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Float32: {
      float v = (float)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    case Dtype::Float64: {
      double v = (double)constScalar;
      std::memcpy(out, &v, sizeof(v));
      out += sizeof(v);
      break;
    }
    default:
      throw std::runtime_error(
          "vkcnn: internal: write_const_one unsupported dtype.");
    }
  };

  // iterate output
  std::vector<std::uint64_t> oIdx(r, 0), sIdx(r, 0);
  std::size_t written = 0;

  auto inc = [&](std::vector<std::uint64_t> &idx,
                 std::span<const std::uint64_t> dims) -> bool {
    if (idx.empty())
      return false;
    size_t ax = idx.size();
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
      const long long n = (long long)inU64[ax];
      const long long bef = (long long)before[ax];
      const long long pos = (long long)oIdx[ax];
      const long long rel = pos - bef;
      if (hostMode == HostPadMode::Constant) {
        if (rel < 0 || rel >= n) {
          outIsConst = true;
          break;
        }
        sIdx[ax] = (std::uint64_t)rel;
      } else if (hostMode == HostPadMode::Edge) {
        sIdx[ax] = (std::uint64_t)clamp(rel, 0, n - 1);
      } else { // Reflect
        sIdx[ax] = (std::uint64_t)reflect_map(rel, n);
      }
    }

    if (dt == Dtype::String) {
      if (hostMode == HostPadMode::Constant && outIsConst) {
        const std::string &s = hasConstVal ? constString : std::string();
        char *cp = (char *)std::malloc(s.size() + 1);
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
        char *cp = (char *)std::malloc(len + 1);
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

} // namespace vkcnn::details
