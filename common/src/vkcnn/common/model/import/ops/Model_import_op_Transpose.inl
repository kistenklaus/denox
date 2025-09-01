#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {

static std::vector<Tensor>
import_op_Transpose(ImportState & /*state*/,
                    std::span<const std::optional<Tensor>> inputs,
                    std::size_t outputCount,
                    const std::unordered_map<std::string, Tensor> &attributes,
                    opset_version /*version*/, const onnx::NodeProto &node) {

  // ---- contract
  if (outputCount != 1) {
    throw std::runtime_error(
        "vkcnn: Transpose must produce exactly one output");
  }
  if (inputs.size() != 1 || !inputs[0].has_value()) {
    throw std::runtime_error("vkcnn: Transpose requires exactly one input");
  }
  const Tensor &X = *inputs[0];

  // ---- reject unsupported kinds
  if (X.isRuntimeTensor())
    throw std::runtime_error(
        "vkcnn: Transpose on runtime tensors is not supported");
  if (X.isString())
    throw std::runtime_error(
        "vkcnn: Transpose on string tensors is not supported");
  if (X.isList())
    throw std::runtime_error(
        "vkcnn: Transpose on list tensors is not supported");
  if (X.isUnknown())
    throw std::runtime_error("vkcnn: Transpose on unknown tensor kind");

  // ---- helper to read perm from attribute (if provided)
  auto read_perm_attr =
      [&](int64_t rank) -> std::optional<std::vector<int64_t>> {
    auto it = attributes.find("perm");
    if (it == attributes.end() || it->second.isUnknown())
      return std::nullopt; // will use default later
    const Tensor &t = it->second;

    std::vector<int64_t> perm;
    if (t.isScalar()) {
      // Perm "should" be a list; tolerate scalar only for rank<=1.
      const auto &s = t.scalar();
      int64_t v;
      switch (s.dtype) {
      case Dtype::Int64:
        v = s.v.i;
        break;
      case Dtype::Int32:
      case Dtype::Int16:
      case Dtype::Int8:
        v = (int64_t)s.v.i;
        break;
      case Dtype::Uint64:
        v = (int64_t)s.v.u;
        break;
      case Dtype::Uint32:
      case Dtype::Uint16:
      case Dtype::Uint8:
        v = (int64_t)s.v.u;
        break;
      default:
        throw std::runtime_error("vkcnn: Transpose: 'perm' must be integer");
      }
      perm.push_back(v);
    } else if (t.isRaw()) {
      const RawTensor &rt = t.raw();
      auto is_int = [](Dtype dt) -> bool {
        switch (dt) {
        case Dtype::Int8:
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
        case Dtype::Uint8:
        case Dtype::Uint16:
        case Dtype::Uint32:
        case Dtype::Uint64:
          return true;
        default:
          return false;
        }
      };
      if (!is_int(rt.type))
        throw std::runtime_error(
            "vkcnn: Transpose: 'perm' dtype must be integer");
      size_t n = 1;
      if (rt.shape.isTensor()) {
        const auto &sv = rt.shape.dims();
        if (sv.size() != 1)
          throw std::runtime_error("vkcnn: Transpose: 'perm' must be 1-D");
        if (!sv[0].isConst())
          throw std::runtime_error(
              "vkcnn: Transpose: 'perm' length must be static");
        n = (size_t)sv[0].value();
      }
      const size_t esz = dtype_size(rt.type);
      if (rt.raw.size() != n * esz)
        throw std::runtime_error(
            "vkcnn: Transpose: 'perm' raw payload size mismatch");
      perm.resize(n);
      const uint8_t *p = (const uint8_t *)rt.raw.data();
      for (size_t i = 0; i < n; ++i) {
        switch (rt.type) {
        case Dtype::Int64:
          perm[i] = ((const int64_t *)p)[i];
          break;
        case Dtype::Int32:
          perm[i] = (int64_t)((const int32_t *)p)[i];
          break;
        case Dtype::Int16:
          perm[i] = (int64_t)((const int16_t *)p)[i];
          break;
        case Dtype::Int8:
          perm[i] = (int64_t)((const int8_t *)p)[i];
          break;
        case Dtype::Uint64:
          perm[i] = (int64_t)((const uint64_t *)p)[i];
          break;
        case Dtype::Uint32:
          perm[i] = (int64_t)((const uint32_t *)p)[i];
          break;
        case Dtype::Uint16:
          perm[i] = (int64_t)((const uint16_t *)p)[i];
          break;
        case Dtype::Uint8:
          perm[i] = (int64_t)((const uint8_t *)p)[i];
          break;
        default:
          break;
        }
      }
    } else {
      throw std::runtime_error("vkcnn: Transpose: 'perm' attribute must be "
                               "scalar or 1-D integer tensor");
    }

    // validate v in [0, rank-1] and uniqueness & length==rank
    if (rank < 0)
      throw std::logic_error("vkcnn: internal rank error");
    if ((int64_t)perm.size() != rank) {
      // Allow rank<=1 special-case: scalar perm 0 for rank 1, empty for rank 0
      if (!(rank <= 1 &&
            (perm.size() == 0 || (perm.size() == 1 && perm[0] == 0)))) {
        throw std::runtime_error(
            "vkcnn: Transpose: 'perm' length must equal rank");
      }
    }
    std::vector<char> seen((size_t)std::max<int64_t>(rank, 1), 0);
    for (size_t i = 0; i < perm.size(); ++i) {
      int64_t a = perm[i];
      if (a < 0 || a >= rank) {
        throw std::runtime_error("vkcnn: Transpose: 'perm' value out of range");
      }
      if (seen[(size_t)a]) {
        throw std::runtime_error("vkcnn: Transpose: 'perm' has duplicate axes");
      }
      seen[(size_t)a] = 1;
    }
    return perm;
  };

  // ---- ScalarTensor: no-op (rank 0); validate perm if provided
  if (X.isScalar()) {
    const int64_t rank = 0;
    auto p = read_perm_attr(rank);
    if (p && !p->empty()) {
      throw std::runtime_error(
          "vkcnn: Transpose: 'perm' must be empty for rank-0 tensors");
    }
    return {X}; // unchanged
  }

  // ---- ShapeTensor-as-data: treat as 1-D vector of Dim; transpose is a no-op
  if (X.isShape()) {
    int64_t rank = 1; // by policy: ShapeTensor is a flat vector (or empty ->
                      // still rank 1 vector len 0)
    // if you prefer to treat truly empty as rank 0, you can change this to:
    // rank = X.shapeTensor().isScalar() ? 0 : 1;
    if (X.shapeTensor().isScalar()) {
      // empty vector: allow empty perm only
      rank = 0;
    }

    auto p = read_perm_attr(rank);
    if (!p.has_value()) {
      // default perm is reverse; for rank<=1 that's a no-op
    } else {
      // validate explicitly: for rank 1, perm must be [0]; for rank 0, empty
      if (rank == 1 && (p->size() != 1 || (*p)[0] != 0)) {
        throw std::runtime_error(
            "vkcnn: Transpose on ShapeTensor (rank-1) requires perm=[0]");
      }
      if (rank == 0 && !p->empty()) {
        throw std::runtime_error(
            "vkcnn: Transpose on ShapeTensor (rank-0) requires perm=[]");
      }
    }
    return {X}; // no-op
  }

  // ---- RawTensor (constant): real axis permutation
  if (X.isRaw()) {
    const RawTensor &rt = X.raw();
    if (rt.type == Dtype::String)
      throw std::runtime_error(
          "vkcnn: Transpose on RawTensor<string> not supported");

    // rank & sizes
    int64_t rank = rt.shape.isScalar() ? 0 : (int64_t)rt.shape.dims().size();
    if (rank <= 1) {
      // scalar or 1-D: transpose is a no-op, but still validate perm
      auto p = read_perm_attr(rank);
      if (p.has_value()) {
        if (rank == 0 && !p->empty())
          throw std::runtime_error(
              "vkcnn: Transpose: 'perm' must be empty for rank-0");
        if (rank == 1 && (p->size() != 1 || (*p)[0] != 0))
          throw std::runtime_error(
              "vkcnn: Transpose: 'perm' must be [0] for rank-1");
      }
      return {X}; // unchanged
    }

    // Read static sizes and ensure all dims are constant (we must reorder
    // bytes)
    std::vector<int64_t> in_sizes((size_t)rank, 0);
    for (int64_t i = 0; i < rank; ++i) {
      const Dim &d = rt.shape.dims()[(size_t)i];
      if (!d.isConst())
        throw std::runtime_error(
            "vkcnn: Transpose RawTensor requires static (constant) shape");
      in_sizes[(size_t)i] = (int64_t)d.value();
    }

    // perm: default is reverse order if not specified
    std::vector<int64_t> perm;
    if (auto p = read_perm_attr(rank)) {
      perm = std::move(*p);
      if (perm.empty() && rank > 0) {
        // if someone passed an empty tensor for rank>0, reject
        throw std::runtime_error(
            "vkcnn: Transpose: 'perm' length must equal rank");
      }
    } else {
      perm.resize((size_t)rank);
      for (int64_t i = 0; i < rank; ++i)
        perm[(size_t)i] = (rank - 1 - i);
    }

    // Validate perm (already mostly done), also compute inverse
    std::vector<int64_t> inv((size_t)rank, -1);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t a = perm[(size_t)i];
      if (a < 0 || a >= rank)
        throw std::runtime_error("vkcnn: Transpose: 'perm' out of range");
      if (inv[(size_t)a] != -1)
        throw std::runtime_error("vkcnn: Transpose: 'perm' duplicates");
      inv[(size_t)a] = i;
    }

    // output sizes
    std::vector<int64_t> out_sizes((size_t)rank, 0);
    for (int64_t i = 0; i < rank; ++i)
      out_sizes[(size_t)i] = in_sizes[(size_t)perm[(size_t)i]];

    // element size & element count
    const size_t esz = dtype_size(rt.type);
    if (esz == 0)
      throw std::runtime_error("vkcnn: Transpose unsupported dtype");
    auto prod = [](const std::vector<int64_t> &v) -> int64_t {
      int64_t p = 1;
      for (auto x : v)
        p *= x;
      return p;
    };
    const int64_t Nelem = prod(in_sizes);
    if ((size_t)Nelem * esz != rt.raw.size()) {
      throw std::runtime_error(
          "vkcnn: Transpose: raw payload size does not match input shape");
    }

    // input strides (row-major)
    std::vector<int64_t> in_strides((size_t)rank, 1);
    for (int64_t d = rank - 2; d >= 0; --d) {
      in_strides[(size_t)d] =
          in_strides[(size_t)(d + 1)] * in_sizes[(size_t)(d + 1)];
    }

    // linearize output -> compute input index:
    // Let o be out multi-index; input index i satisfies: i[perm[d]] = o[d]
    // So i[j] = o[inv[j]]
    auto lin_to_multi = [&](int64_t lin) -> std::vector<int64_t> {
      std::vector<int64_t> idx((size_t)rank, 0);
      for (int64_t d = rank - 1; d >= 0; --d) {
        int64_t sd = out_sizes[(size_t)d];
        idx[(size_t)d] = sd == 0 ? 0 : (lin % sd);
        if (sd != 0)
          lin /= sd;
      }
      return idx;
    };

    RawTensor out;
    out.type = rt.type;

    // build output ShapeTensor
    ShapeVector out_dims;
    out_dims.reserve((size_t)rank);
    for (int64_t d = 0; d < rank; ++d) {
      out_dims.push_back(Dim::Const((uint64_t)out_sizes[(size_t)d]));
    }
    out.shape = ShapeTensor::Tensor(std::move(out_dims));
    out.raw.resize((size_t)Nelem * esz);

    if (Nelem == 0) {
      return {Tensor::Raw(std::move(out))};
    }

    const uint8_t *src = reinterpret_cast<const uint8_t *>(rt.raw.data());
    uint8_t *dst = reinterpret_cast<uint8_t *>(out.raw.data());

    for (int64_t lin = 0; lin < Nelem; ++lin) {
      std::vector<int64_t> oidx = lin_to_multi(lin);
      // compute input linear offset
      int64_t in_lin = 0;
      for (int64_t j = 0; j < rank; ++j) {
        int64_t i_j = oidx[(size_t)inv[(size_t)j]];
        in_lin += i_j * in_strides[(size_t)j];
      }
      std::memcpy(dst + (size_t)lin * esz, src + (size_t)in_lin * esz, esz);
    }

    return {Tensor::Raw(std::move(out))};
  }

  // Anything else should have been covered
  throw std::runtime_error(fmt::format(
      "vkcnn: Transpose not supported for given input kind (node='{}')",
      node.name()));
}

} // namespace vkcnn::details
