#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor> import_op_Reshape(
    ImportState & /*state*/, std::span<const std::optional<Tensor>> inputs,
    std::size_t outputCount,
    const std::unordered_map<std::string, Attribute> &attributes,
    [[maybe_unused]] opset_version /*version*/, const onnx::NodeProto &node) {

  // Arity
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\" expects 2 inputs (data, shape).", node.name()));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\" must have exactly 1 output.", node.name()));

  const Tensor &dataT = *inputs[0];
  const Tensor &shapeT = *inputs[1];

  // Device tensors unsupported
  if (dataT.isDevice() || shapeT.isDevice())
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\": runtime tensors not supported.", node.name()));

  const HostTensor &data = dataT.host();
  const HostTensor &shape = shapeT.host();

  // Require static input shape (HostTensor::reshape needs static)
  if (!data.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\": input shape must be static.", node.name()));

  // Shape tensor must be 1-D (ONNX spec)
  if (!shape.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\": shape tensor must be constant.", node.name()));
  const auto shpU64 = shape.shape().toU64();
  if (shpU64.size() != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Reshape \"{}\": shape tensor must be 1-D.", node.name()));

  // Parse 'allowzero' (default 0: zeros mean copy from input dim at same index)
  bool allowZero = false;
  if (auto it = attributes.find("allowzero"); it != attributes.end()) {
    if (!it->second.isInt())
      throw std::runtime_error(fmt::format(
          "vkcnn: Reshape \"{}\": attribute 'allowzero' must be int.",
          node.name()));
    allowZero = (it->second.i() != 0);
  }

  // Read shape values (require INT64 or SYM-with-constant)
  std::vector<std::int64_t>
      req; // requested dims with -1/0 semantics before expansion
  {
    if (shape.type() == Dtype::Int64) {
      auto sv = shape.storage()->i64();
      req.assign(sv.begin(), sv.end());
    } else if (shape.type() == Dtype::Sym) {
      // Allow Sym only if all entries are constant
      auto sv = shape.storage()->sym();
      req.resize(sv.size());
      for (std::size_t i = 0; i < sv.size(); ++i) {
        // You said SYM storage is trivial; require constants here
        // (If you want to support symbolic reshape target, you'd need a
        // different path) For now keep it simple:
        throw std::runtime_error(fmt::format(
            "vkcnn: Reshape \"{}\": symbolic shape entries are not supported.",
            node.name()));
      }
    } else {
      throw std::runtime_error(fmt::format(
          "vkcnn: Reshape \"{}\": shape tensor must be INT64.", node.name()));
    }
  }

  // Build target dims (Symbolic constants), applying -1 and zero rules
  const auto inDimsU64 = data.shape().toU64();
  const std::size_t inRank = inDimsU64.size();

  // If zeros are used with allowzero==0, target rank must match input rank
  // (copy same index).
  if (!allowZero) {
    bool hasZero = false;
    for (auto v : req)
      if (v == 0) {
        hasZero = true;
        break;
      }
    if (hasZero && req.size() != inRank) {
      throw std::runtime_error(
          fmt::format("vkcnn: Reshape \"{}\": zeros in target shape require "
                      "same rank as input.",
                      node.name()));
    }
  }

  // Count -1 positions and validate negatives
  int inferPos = -1;
  for (std::size_t i = 0; i < req.size(); ++i) {
    const auto v = req[i];
    if (v < 0) {
      if (v == -1) {
        if (inferPos >= 0) {
          throw std::runtime_error(fmt::format(
              "vkcnn: Reshape \"{}\": only one -1 is allowed in shape.",
              node.name()));
        }
        inferPos = static_cast<int>(i);
      } else {
        throw std::runtime_error(
            fmt::format("vkcnn: Reshape \"{}\": invalid negative dimension {}.",
                        node.name(), v));
      }
    }
  }

  // Compute product of known requested dims (after zero-copy expansion if
  // needed)
  std::vector<std::uint64_t>
      targetDims; // fully expanded positive (or zero if allowZero)
  targetDims.reserve(req.size());

  // Expand zeros (if !allowZero) and validate non-negatives
  for (std::size_t i = 0; i < req.size(); ++i) {
    const auto v = req[i];
    if (v == 0 && !allowZero) {
      // copy from input dim at same index
      targetDims.push_back((i < inRank) ? inDimsU64[i] : 0ULL);
    } else if (v == -1) {
      // placeholder; will fill later
      targetDims.push_back(0ULL);
    } else {
      // v >= 0 (including zero when allowZero==true)
      targetDims.push_back(static_cast<std::uint64_t>(v));
    }
  }

  // Infer -1 if present
  if (inferPos >= 0) {
    // oldNumel
    std::size_t oldN = 1;
    for (auto d : inDimsU64)
      oldN *= static_cast<std::size_t>(d);

    // product of specified target dims (treat 0 as 0; if productKnown==0,
    // inference is ambiguous)
    std::size_t knownProd = 1;
    bool hasZeroDim = false;
    for (std::size_t i = 0; i < targetDims.size(); ++i) {
      if ((int)i == inferPos)
        continue;
      if (targetDims[i] == 0ULL) {
        hasZeroDim = true;
      }
      knownProd *= static_cast<std::size_t>(targetDims[i]);
    }

    if (hasZeroDim) {
      // Ambiguous: -1 with zero dims — disallow to keep semantics simple
      throw std::runtime_error(
          fmt::format("vkcnn: Reshape \"{}\": cannot infer -1 when other "
                      "target dims are zero.",
                      node.name()));
    }
    if (knownProd == 0) {
      throw std::runtime_error(
          fmt::format("vkcnn: Reshape \"{}\": cannot infer -1 with zero "
                      "product of known dims.",
                      node.name()));
    }
    if (oldN % knownProd != 0) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Reshape \"{}\": element count mismatch (cannot infer -1).",
          node.name()));
    }
    const std::size_t inferred = oldN / knownProd;
    targetDims[static_cast<std::size_t>(inferPos)] =
        static_cast<std::uint64_t>(inferred);
  }

  // Build TensorShape from targetDims (all constants now)
  std::vector<Symbolic> outSyms;
  outSyms.reserve(targetDims.size());
  const auto g = data.shape().graph();
  for (auto d : targetDims) {
    outSyms.emplace_back(g, Sym::Const(static_cast<std::int64_t>(d)));
  }
  TensorShape newShape{g, std::move(outSyms)};

  // Use HostTensor::reshape helper (will check numel equality and contiguity)
  HostTensor out = data.reshape(newShape);
  return {Tensor::Host(std::move(out))};
}

} // namespace vkcnn::details
