#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Gather(
    ImportState &state, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {

  // Arity
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(
        fmt::format("vkcnn: Gather \"{}\" expects 2 inputs.", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Gather \"{}\" must have exactly 1 output.", node.name()));

  // Host-only
  const Tensor &dataT = *inputs[0];
  const Tensor &indicesT = *inputs[1];
  if (dataT.isDevice() || indicesT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Gather \"{}\": runtime tensors not supported.", node.name()));
  const HostTensor &data = dataT.host();
  const HostTensor &indices = indicesT.host();

  // axis
  std::int64_t axis = 0;
  if (auto it = attributes.find("axis"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: Gather \"{}\": attribute 'axis' must be int.", node.name()));
    axis = it->second.i();
  }

  // Must be static (host op)
  if (!data.isConstant() || !indices.isConstant())
    throw std::runtime_error(
        fmt::format("vkcnn: Gather \"{}\": dynamic host tensors unsupported.",
                    node.name()));
  // We’ll use constIndexOf → require constant view too.
  if (!data.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Gather \"{}\": non-constant view unsupported.", node.name()));

  // Normalize axis
  const auto dRank = static_cast<std::int64_t>(data.rank());
  if (dRank <= 0)
    throw std::runtime_error(fmt::format(
        "vkcnn: Gather \"{}\": data rank must be >=1.", node.name()));
  if (axis < 0)
    axis += dRank;
  if (axis < 0 || axis >= dRank)
    throw std::runtime_error(
        fmt::format("vkcnn: Gather \"{}\": axis {} out of range for rank {}.",
                    node.name(), axis, dRank));

  // Indices dtype
  if (indices.type() != Dtype::Int64)
    throw std::runtime_error(fmt::format(
        "vkcnn: Gather \"{}\": indices must be INT64.", node.name()));

  // Make indices contiguous for easy reading
  HostTensor idxC = indices.contiguous();
  const auto idxDimsU64 = idxC.shape().toU64();
  const size_t idxRank = idxDimsU64.size();

  // Fast path: scalar index → select
  if (idxRank == 0) {
    const auto sv = idxC.storage()->i64();
    const std::int64_t v = sv.empty() ? 0 : sv[0];

    const auto dataDims = data.shape().toU64();
    const auto ax = static_cast<size_t>(axis);
    if (v < 0 || static_cast<std::uint64_t>(v) >= dataDims[ax])
      throw std::runtime_error(fmt::format(
          "vkcnn: Gather \"{}\": index {} out of bounds for axis {}.",
          node.name(), v, axis));

    HostTensor out = data.select(ax, static_cast<std::uint64_t>(v));
    return {Tensor::Host(std::move(out))};
  }

  // Fast path: 1-D arange → narrow
  if (idxRank == 1) {
    const auto sv = idxC.storage()->i64();
    const size_t L = sv.size();
    if (L == 0) {
      HostTensor out = data.narrow(static_cast<size_t>(axis), 0, 0);
      return {Tensor::Host(std::move(out))};
    }
    const std::int64_t start = sv[0];
    bool is_arange = (start >= 0);
    for (size_t i = 1; is_arange && i < L; ++i) {
      if (sv[i] != start + static_cast<std::int64_t>(i))
        is_arange = false;
      if (sv[i] < 0)
        is_arange = false;
    }
    if (is_arange) {
      const auto dataDims = data.shape().toU64();
      const auto ax = static_cast<size_t>(axis);
      const auto axSize = dataDims[ax];
      const std::uint64_t ustart = static_cast<std::uint64_t>(start);
      if (ustart > axSize || (L > 0 && ustart + (L - 1) >= axSize))
        throw std::runtime_error(fmt::format(
            "vkcnn: Gather \"{}\": range [{}, {}] out of bounds on axis {}.",
            node.name(), start, start + static_cast<std::int64_t>(L - 1),
            axis));

      HostTensor out = data.narrow(ax, ustart, L);
      return {Tensor::Host(std::move(out))};
    }
  }

  // General path: materialize
  const auto dataDims = data.shape().toU64();
  const auto idxDims = idxDimsU64;

  // Output shape = replace data[axis] with indices.shape
  std::vector<Symbolic> outSyms;
  outSyms.reserve(dataDims.size() - 1 + idxDims.size());
  {
    const auto g = data.shape().graph();
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
      outSyms.emplace_back(g,
                           Sym::Const(static_cast<std::int64_t>(dataDims[i])));
    for (auto d : idxDims)
      outSyms.emplace_back(g, Sym::Const(static_cast<std::int64_t>(d)));
    for (size_t i = static_cast<size_t>(axis) + 1; i < dataDims.size(); ++i)
      outSyms.emplace_back(g,
                           Sym::Const(static_cast<std::int64_t>(dataDims[i])));
  }
  TensorShape outShape{data.shape().graph(), std::move(outSyms)};
  const auto outDims = outShape.toU64();

  const Dtype dt = data.type();
  const size_t elem = data.elemSize();

  std::size_t outCount = 1;
  for (auto d : outDims)
    outCount *= static_cast<std::size_t>(d);

  // Allocate destination
  std::shared_ptr<HostTensorStorage> dstStore;
  if (dt == Dtype::String) {
    auto table = static_cast<char **>(std::malloc(outCount * sizeof(char *)));
    if (!table)
      throw std::bad_alloc();
    dstStore =
        std::make_shared<HostTensorStorage>(HostTensorStorage::TakeOwnership(
            Dtype::String, table, outCount * sizeof(char *)));
  } else {
    void *raw = std::malloc(outCount * elem);
    if (!raw)
      throw std::bad_alloc();
    dstStore = std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, outCount * elem));
  }

  // Prepare iteration
  const size_t dRankZ = dataDims.size();
  const size_t iRankZ = idxDims.size();
  const size_t oRankZ = outDims.size();

  const auto &view = data.view(); // constant by check above
  const auto *srcBase = static_cast<const std::byte *>(data.storage()->data());
  const auto *idxBase = idxC.storage()->i64().data();

  std::vector<std::uint64_t> oIdx(oRankZ, 0);
  std::vector<std::uint64_t> dIdx(dRankZ, 0);
  std::vector<std::uint64_t> iIdx(iRankZ, 0);

  auto inc = [](std::vector<std::uint64_t> &idx,
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

  auto idxLinear = [&](std::span<const std::uint64_t> dims) -> std::size_t {
    if (iRankZ == 0)
      return 0;
    std::size_t lin = 0, stride = 1;
    for (size_t k = 0; k < iRankZ; ++k) {
      size_t ax = iRankZ - 1 - k;
      lin += static_cast<std::size_t>(iIdx[ax] * stride);
      stride *= static_cast<std::size_t>(dims[ax]);
    }
    return lin;
  };

  std::byte *dstBytes = (dt == Dtype::String)
                            ? nullptr
                            : static_cast<std::byte *>(dstStore->data());
  char **dstStrs =
      (dt == Dtype::String) ? static_cast<char **>(dstStore->data()) : nullptr;
  std::size_t outWritten = 0;

  while (true) {
    // oIdx → (iIdx, dIdx)
    size_t od = 0;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i, ++od)
      dIdx[i] = oIdx[od];

    for (size_t i = 0; i < iRankZ; ++i, ++od)
      iIdx[i] = oIdx[od];

    std::int64_t selectVal =
        (iRankZ == 0) ? idxBase[0] : idxBase[idxLinear(idxDims)];
    const auto ax = static_cast<size_t>(axis);
    if (selectVal < 0 || static_cast<std::uint64_t>(selectVal) >= dataDims[ax])
      throw std::runtime_error(fmt::format(
          "vkcnn: Gather \"{}\": index {} out of bounds on axis {}.",
          node.name(), selectVal, axis));
    dIdx[ax] = static_cast<std::uint64_t>(selectVal);

    for (size_t i = ax + 1; i < dRankZ; ++i, ++od)
      dIdx[i] = oIdx[od];

    // Source element index (in elements), then copy
    const std::size_t srcElem = view.constIndexOf(
        std::span<const std::uint64_t>(dIdx.data(), dIdx.size()));

    if (dt == Dtype::String) {
      const char *srcp =
          reinterpret_cast<char *const *>(data.storage()->data())[srcElem];
      const std::size_t len = std::strlen(srcp);
      char *copy = static_cast<char *>(std::malloc(len + 1));
      if (!copy)
        throw std::bad_alloc();
      std::memcpy(copy, srcp, len + 1);
      dstStrs[outWritten++] = copy;
    } else {
      const std::size_t srcByte = srcElem * elem;
      std::memcpy(dstBytes, srcBase + srcByte, elem);
      dstBytes += elem;
      ++outWritten;
    }

    if (!inc(oIdx, outDims))
      break;
  }

  HostTensor outHT(outShape, std::move(dstStore));
  return {Tensor::Host(std::move(outHT))};
}

} // namespace vkcnn::details
