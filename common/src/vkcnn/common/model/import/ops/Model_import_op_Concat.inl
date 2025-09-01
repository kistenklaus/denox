#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Concat(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Concat \"{}\" must have exactly 1 output.", node.name()));
  if (inputs.empty())
    throw std::runtime_error(fmt::format(
        "vkcnn: Concat \"{}\" needs at least 1 input.", node.name()));
  for (size_t i = 0; i < inputs.size(); ++i)
    if (!inputs[i].has_value())
      throw std::runtime_error(fmt::format(
          "vkcnn: Concat \"{}\": input {} is missing.", node.name(), i));

  // axis (default 0)
  std::int64_t axis = 0;
  if (auto it = attributes.find("axis"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: Concat \"{}\": attribute 'axis' must be int.", node.name()));
    axis = it->second.i();
  }

  // ---------- Device path (unchanged): exactly 2 device tensors, channel axis
  // ----------
  {
    bool allDevice = true, anyDevice = false;
    for (const auto &topt : inputs) {
      anyDevice = anyDevice || topt->isDevice();
      allDevice = allDevice && topt->isDevice();
    }
    if (anyDevice && !allDevice)
      throw std::runtime_error(fmt::format(
          "vkcnn: Concat \"{}\": cannot mix device and host tensors.",
          node.name()));

    if (allDevice) {
      if (inputs.size() != 2)
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": device concat supports exactly 2 inputs.",
            node.name()));

      const DeviceTensor &d0 = inputs[0]->device();
      const DeviceTensor &d1 = inputs[1]->device();

      const auto s0 = d0.shape();
      const auto s1 = d1.shape();
      const size_t r0 = s0.rank(), r1 = s1.rank();
      if ((r0 != 3 && r0 != 4) || r1 != r0)
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": device tensors must be rank 3/4 and match.",
            node.name()));

      const size_t chAxis = (r0 == 4) ? 1u : 0u;
      std::int64_t naxis = axis;
      if (naxis < 0)
        naxis += static_cast<std::int64_t>(r0);
      if (naxis < 0 || naxis >= static_cast<std::int64_t>(r0))
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": axis {} out of range for rank {}.",
            node.name(), axis, r0));
      if (static_cast<size_t>(naxis) != chAxis)
        throw std::runtime_error(
            fmt::format("vkcnn: Concat \"{}\": device concat supported only on "
                        "channel axis {}.",
                        node.name(), chAxis));

      const size_t axH = (r0 == 4) ? 2u : 1u;
      const size_t axW = (r0 == 4) ? 3u : 2u;
      if (!(s0[axH] == s1[axH]) || !(s0[axW] == s1[axW]))
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": spatial dims must match for device concat.",
            node.name()));

      vkcnn::Tensor outHandle = state.output.concat(d0.handle(), d1.handle());
      DeviceTensor outDev(r0, std::move(outHandle));
      return {Tensor::Device(std::move(outDev))};
    }
  }

  // ---------- Host path (variadic) ----------
  std::vector<const HostTensor *> hs;
  hs.reserve(inputs.size());
  for (const auto &topt : inputs) {
    if (!topt->isHost())
      throw std::runtime_error(fmt::format(
          "vkcnn: Concat \"{}\": expected all host tensors.", node.name()));
    hs.push_back(&topt->host());
  }

  // All static & same rank
  if (!hs[0]->isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Concat \"{}\": dynamic host tensors unsupported.",
                    node.name()));
  const auto d0 = hs[0]->shape().toU64();
  const size_t R = d0.size();
  if (R == 0)
    throw std::runtime_error(fmt::format(
        "vkcnn: Concat \"{}\": cannot concat scalars.", node.name()));
  for (size_t i = 1; i < hs.size(); ++i) {
    if (!hs[i]->isConstant())
      throw std::runtime_error(
          fmt::format("vkcnn: Concat \"{}\": dynamic host tensors unsupported.",
                      node.name()));
    const auto di = hs[i]->shape().toU64();
    if (di.size() != R)
      throw std::runtime_error(fmt::format(
          "vkcnn: Concat \"{}\": rank mismatch at input {}.", node.name(), i));
  }

  // Determine common output dtype:
  // - if all the same → keep
  // - else if all in {Int64, Sym} → Sym
  // - else → error
  auto dt0 = hs[0]->type();
  bool allSame = true;
  bool onlyInt64OrSym = (dt0 == Dtype::Int64 || dt0 == Dtype::Sym);
  for (size_t i = 1; i < hs.size(); ++i) {
    auto dti = hs[i]->type();
    allSame = allSame && (dti == dt0);
    onlyInt64OrSym =
        onlyInt64OrSym && (dti == Dtype::Int64 || dti == Dtype::Sym);
  }

  Dtype outDt;
  if (allSame) {
    outDt = dt0;
  } else if (onlyInt64OrSym) {
    outDt = Dtype::Sym; // promote
  } else {
    // keep the old, strict error:
    // report where mismatch happened for clarity
    for (size_t i = 1; i < hs.size(); ++i) {
      if (hs[i]->type() != dt0) {
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": dtype mismatch ({} vs {}) at input {}.",
            node.name(), (int)dt0, (int)hs[i]->type(), i));
      }
    }
    // fallback (shouldn’t reach)
    outDt = dt0;
  }

  // Normalize axis
  std::int64_t naxis = axis;
  if (naxis < 0)
    naxis += static_cast<std::int64_t>(R);
  if (naxis < 0 || naxis >= static_cast<std::int64_t>(R))
    throw std::runtime_error(
        fmt::format("vkcnn: Concat \"{}\": axis {} out of range for rank {}.",
                    node.name(), axis, R));
  const size_t A = static_cast<size_t>(naxis);

  // Shape compatibility & out shape
  auto g = hs[0]->shape().graph();
  std::vector<std::uint64_t> outU64 = d0;
  std::uint64_t sumAlongA = d0[A];
  for (size_t i = 1; i < hs.size(); ++i) {
    const auto di = hs[i]->shape().toU64();
    for (size_t ax = 0; ax < R; ++ax) {
      if (ax == A)
        continue;
      if (di[ax] != d0[ax])
        throw std::runtime_error(
            fmt::format("vkcnn: Concat \"{}\": non-concat dimension {} "
                        "mismatch at input {}.",
                        node.name(), ax, i));
    }
    sumAlongA += di[A];
  }
  outU64[A] = sumAlongA;

  std::vector<Symbolic> outSyms;
  outSyms.reserve(R);
  for (size_t ax = 0; ax < R; ++ax)
    outSyms.emplace_back(g, Sym::Const(static_cast<std::int64_t>(outU64[ax])));
  TensorShape outShape{g, std::move(outSyms)};

  // Early return for empty
  auto prod = [](const std::vector<std::uint64_t> &v, size_t from, size_t to) {
    std::size_t p = 1;
    for (size_t i = from; i < to; ++i)
      p *= (size_t)v[i];
    return p;
  };
  const std::size_t outerCount = prod(outU64, 0, A);
  const std::size_t innerElems = prod(outU64, A + 1, R);
  std::size_t totalElems = outerCount * innerElems * (size_t)outU64[A];
  if (totalElems == 0) {
    std::shared_ptr<HostTensorStorage> store =
        (outDt == Dtype::String)
            ? std::make_shared<HostTensorStorage>(
                  HostTensorStorage::TakeOwnership(Dtype::String, nullptr, 0))
            : std::make_shared<HostTensorStorage>(
                  HostTensorStorage::Raw(outDt, nullptr, 0));
    return {Tensor::Host(HostTensor(outShape, std::move(store)))};
  }

  // Make inputs contiguous
  std::vector<HostTensor> contig;
  contig.reserve(hs.size());
  for (auto *h : hs)
    contig.push_back(h->contiguous());

  // Per-input extent along axis A
  std::vector<std::size_t> lenA(hs.size());
  for (size_t i = 0; i < hs.size(); ++i)
    lenA[i] = (size_t)contig[i].shape().toU64()[A];

  // Handle strings: must all be strings if any is string
  if (outDt == Dtype::String) {
    for (size_t i = 0; i < contig.size(); ++i)
      if (contig[i].type() != Dtype::String)
        throw std::runtime_error(fmt::format(
            "vkcnn: Concat \"{}\": cannot concat String with non-String.",
            node.name()));

    auto table = static_cast<char **>(std::malloc(totalElems * sizeof(char *)));
    if (!table)
      throw std::bad_alloc();
    auto dst =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::String, table, totalElems * sizeof(char *)));

    std::size_t outIndex = 0;
    for (std::size_t o = 0; o < outerCount; ++o) {
      for (size_t i = 0; i < contig.size(); ++i) {
        const std::size_t chunkElems = lenA[i] * innerElems;
        const char *const *stab =
            reinterpret_cast<const char *const *>(contig[i].storage()->data());
        const std::size_t base = o * chunkElems;
        for (std::size_t k = 0; k < chunkElems; ++k) {
          const char *src = stab[base + k];
          const std::size_t L = std::strlen(src);
          char *cp = static_cast<char *>(std::malloc(L + 1));
          if (!cp)
            throw std::bad_alloc();
          std::memcpy(cp, src, L + 1);
          table[outIndex++] = cp;
        }
      }
    }

    return {Tensor::Host(HostTensor(outShape, std::move(dst)))};
  }

  // Non-string path.
  // If outDt == Sym, we may need to convert Int64 chunks → Sym on the fly.
  const bool promoteToSym = (outDt == Dtype::Sym);

  if (!promoteToSym) {
    // All inputs must already share the same dtype (= outDt)
    for (size_t i = 0; i < contig.size(); ++i)
      if (contig[i].type() != outDt)
        throw std::runtime_error(
            fmt::format("vkcnn: Concat \"{}\": dtype mismatch (cannot "
                        "reconcile to {} at input {}).",
                        node.name(), (int)outDt, i));

    const std::size_t elem = contig[0].elemSize();
    void *raw = std::malloc(totalElems * elem);
    if (!raw)
      throw std::bad_alloc();
    auto dst = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(outDt, raw, totalElems * elem));

    const std::size_t innerBytes = innerElems * elem;
    std::byte *dbytes = static_cast<std::byte *>(dst->data());

    for (std::size_t o = 0; o < outerCount; ++o) {
      std::size_t doff = o * innerBytes * (size_t)outU64[A];
      for (size_t i = 0; i < contig.size(); ++i) {
        const std::size_t chunkBytes = lenA[i] * innerBytes;
        const std::byte *s =
            static_cast<const std::byte *>(contig[i].storage()->data());
        const std::size_t soff = o * chunkBytes;
        std::memcpy(dbytes + doff, s + soff, chunkBytes);
        doff += chunkBytes;
      }
    }

    return {Tensor::Host(HostTensor(outShape, std::move(dst)))};
  }

  // promote {Int64, Sym} → Sym
  {
    const std::size_t elemOut = sizeof(Sym);
    void *raw = std::malloc(totalElems * elemOut);
    if (!raw)
      throw std::bad_alloc();
    auto dst =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::Sym, raw, totalElems * elemOut));

    Sym *out = static_cast<Sym *>(dst->data());
    std::size_t outIndex = 0;

    for (std::size_t o = 0; o < outerCount; ++o) {
      for (size_t i = 0; i < contig.size(); ++i) {
        const std::size_t chunkElems = lenA[i] * innerElems;
        const Dtype idt = contig[i].type();

        if (idt == Dtype::Sym) {
          const Sym *src = contig[i].storage()->sym().data();
          const std::size_t base = o * chunkElems;
          std::memcpy(out + outIndex, src + base, chunkElems * sizeof(Sym));
          outIndex += chunkElems;
        } else if (idt == Dtype::Int64) {
          const std::int64_t *src = contig[i].storage()->i64().data();
          const std::size_t base = o * chunkElems;
          for (std::size_t k = 0; k < chunkElems; ++k)
            out[outIndex++] = Sym::Const(src[base + k]);
        } else {
          throw std::runtime_error(fmt::format(
              "vkcnn: Concat \"{}\": unexpected dtype in Sym-promotion path.",
              node.name()));
        }
      }
    }

    return {Tensor::Host(HostTensor(outShape, std::move(dst)))};
  }
}
} // namespace vkcnn::details
