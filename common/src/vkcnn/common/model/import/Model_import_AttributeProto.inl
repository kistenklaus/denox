#pragma once

#include "vkcnn/common/model/import/Model_import_TensorProto.inl"
#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <stdexcept>
#include <utility>
namespace vkcnn::details {

static onnx::AttributeProto_AttributeType
infer_attribute_type_loose(const onnx::AttributeProto &a,
                           std::string_view nodeName) {
  using AT = onnx::AttributeProto_AttributeType;

  const std::string aname =
      a.name().empty() ? std::string("<unnamed>") : a.name();

  // Not supported until you implement function-attribute indirection.
  if (!a.ref_attr_name().empty()) {
    throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                             "\" attribute \"" + aname +
                             "\": ref_attr_name is unsupported");
  }

  // If exporter provided an explicit type, trust it — but sanity check payload
  // presence for complex families so callers can safely access fields.
  auto explicit_t = a.type();
  if (explicit_t != AT::AttributeProto_AttributeType_UNDEFINED) {
    // Minimal payload checks to uphold the "safe to access" invariant.
    switch (explicit_t) {
    case AT::AttributeProto_AttributeType_TENSOR:
      if (!a.has_t())
        throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=TENSOR but no tensor payload");
      break;
    case AT::AttributeProto_AttributeType_SPARSE_TENSOR:
      if (!a.has_sparse_tensor())
        throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=SPARSE_TENSOR but no payload");
      break;
    case AT::AttributeProto_AttributeType_GRAPH:
      if (!a.has_g())
        throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=GRAPH but no graph payload");
      break;
    case AT::AttributeProto_AttributeType_TYPE_PROTO:
      if (!a.has_tp())
        throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=TYPE_PROTO but no payload");
      break;
    case AT::AttributeProto_AttributeType_TENSORS:
      // empty list is still a valid payload; safe to access tensors()
      break;
    case AT::AttributeProto_AttributeType_SPARSE_TENSORS:
      break;
    case AT::AttributeProto_AttributeType_GRAPHS:
      break;
    case AT::AttributeProto_AttributeType_TYPE_PROTOS:
      break;
    case AT::AttributeProto_AttributeType_FLOATS:
    case AT::AttributeProto_AttributeType_INTS:
    case AT::AttributeProto_AttributeType_STRINGS:
      // empty lists are allowed; accessors are safe
      break;
    case AT::AttributeProto_AttributeType_FLOAT:
    case AT::AttributeProto_AttributeType_INT:
    case AT::AttributeProto_AttributeType_STRING:
      // Scalars are always accessible in proto3 (may be defaulted).
      break;
    case AT::AttributeProto_AttributeType_UNDEFINED:
    default:
      break;
    }
    return explicit_t;
  }

  // ---------- Implicit inference path ----------
  auto pick_only = [&](bool cond, AT t, AT &acc, bool &found) {
    if (!cond)
      return;
    if (found && acc != t) {
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\" mixes multiple payload families");
    }
    found = true;
    acc = t;
  };

  AT ret = AT::AttributeProto_AttributeType_UNDEFINED;
  bool found = false;

  // Singular complex (unambiguous)
  pick_only(a.has_t(), AT::AttributeProto_AttributeType_TENSOR, ret, found);
  pick_only(a.has_g(), AT::AttributeProto_AttributeType_GRAPH, ret, found);
  pick_only(a.has_sparse_tensor(),
            AT::AttributeProto_AttributeType_SPARSE_TENSOR, ret, found);
  pick_only(a.has_tp(), AT::AttributeProto_AttributeType_TYPE_PROTO, ret,
            found);

  // Plural complex (unambiguous; allow empty as "present" only for explicit
  // types, so here we only infer when non-empty)
  pick_only(a.tensors_size() > 0, AT::AttributeProto_AttributeType_TENSORS, ret,
            found);
  pick_only(a.graphs_size() > 0, AT::AttributeProto_AttributeType_GRAPHS, ret,
            found);
  pick_only(a.sparse_tensors_size() > 0,
            AT::AttributeProto_AttributeType_SPARSE_TENSORS, ret, found);
  pick_only(a.type_protos_size() > 0,
            AT::AttributeProto_AttributeType_TYPE_PROTOS, ret, found);

  // Lists of scalars (infer only if non-empty)
  pick_only(a.floats_size() > 0, AT::AttributeProto_AttributeType_FLOATS, ret,
            found);
  pick_only(a.ints_size() > 0, AT::AttributeProto_AttributeType_INTS, ret,
            found);
  pick_only(a.strings_size() > 0, AT::AttributeProto_AttributeType_STRINGS, ret,
            found);

  // Scalars: infer only on non-default values to avoid proto3 presence trap.
  if (!found) {
    const bool has_i = (a.i() != 0);
    const bool has_f = (a.f() != 0.0f);
    const bool has_s = !a.s().empty();

    const int count = (has_i ? 1 : 0) + (has_f ? 1 : 0) + (has_s ? 1 : 0);
    if (count > 1) {
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\" has conflicting scalar payloads");
    }
    if (count == 1) {
      ret = has_i   ? AT::AttributeProto_AttributeType_INT
            : has_f ? AT::AttributeProto_AttributeType_FLOAT
                    : AT::AttributeProto_AttributeType_STRING;
      found = true;
    }
  }

  // Post-validate chosen family so the caller can safely access fields.
  switch (ret) {
  case AT::AttributeProto_AttributeType_TENSOR:
    if (!a.has_t())
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred TENSOR but no tensor payload");
    break;
  case AT::AttributeProto_AttributeType_GRAPH:
    if (!a.has_g())
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred GRAPH but no graph payload");
    break;
  case AT::AttributeProto_AttributeType_SPARSE_TENSOR:
    if (!a.has_sparse_tensor())
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred SPARSE_TENSOR but no payload");
    break;
  case AT::AttributeProto_AttributeType_TYPE_PROTO:
    if (!a.has_tp())
      throw std::runtime_error("vkcnn: node \"" + std::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred TYPE_PROTO but no payload");
    break;
  case AT::AttributeProto_AttributeType_TENSORS:
    if (a.tensors_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_GRAPHS:
    if (a.graphs_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_SPARSE_TENSORS:
    if (a.sparse_tensors_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_TYPE_PROTOS:
    if (a.type_protos_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_FLOATS:
    if (a.floats_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_INTS:
    if (a.ints_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_STRINGS:
    if (a.strings_size() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_FLOAT:
    if (a.f() == 0.0f)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_INT:
    if (a.i() == 0)
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_STRING:
    if (a.s().empty())
      return AT::AttributeProto_AttributeType_UNDEFINED;
    break;
  case AT::AttributeProto_AttributeType_UNDEFINED:
  default:
    break;
  }

  return ret; // may still be UNDEFINED → caller makes a None tensor.
}

static std::pair<std::string, Tensor>
parse_attribute(const ImportState &state, const onnx::AttributeProto &attrib,
                std::string_view nodeName) {
  const std::string &name = attrib.name();

  if (!attrib.ref_attr_name().empty()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} is invalid. vkcnn does not "
                    "support reference attributes. Normally found with "
                    "functions definitions, which are not supported.",
                    nodeName, name));
    ;
  }
  auto type = infer_attribute_type_loose(attrib, nodeName);

  switch (type) {
  case onnx::AttributeProto_AttributeType_UNDEFINED:
    // NOTE: Does not have to be a scalar, just a placeholder here.
    return std::make_pair(attrib.name(), Tensor::Unknown());
  case onnx::AttributeProto_AttributeType_FLOAT:
    return std::make_pair(attrib.name(), Tensor::Scalar(attrib.f()));
  case onnx::AttributeProto_AttributeType_INT:
    return std::make_pair(attrib.name(), Tensor::Scalar(attrib.i()));
  case onnx::AttributeProto_AttributeType_STRING:
    return std::make_pair(attrib.name(), Tensor::String(attrib.s()));
  case onnx::AttributeProto_AttributeType_TENSOR:
    return std::make_pair(attrib.name(), parse_tensor(state, attrib.t()));
  case onnx::AttributeProto_AttributeType_GRAPH:
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} invalid. Subgraph attributes "
                    "are not supported.",
                    nodeName, name));
  case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} invalid. Sparse tensor "
                    "attributes are not supported.",
                    nodeName, name));
  case onnx::AttributeProto_AttributeType_TYPE_PROTO:
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} invalid. Type prototype "
                    "attributes are not supported.",
                    nodeName, name));
  case onnx::AttributeProto_AttributeType_FLOATS: {
    std::vector<Tensor> floats;
    floats.reserve(attrib.floats_size());
    for (const auto &f : attrib.floats()) {
      floats.push_back(Tensor::Scalar(f));
    }
    return std::make_pair(attrib.name(), Tensor::List(std::move(floats)));
  }
  case onnx::AttributeProto_AttributeType_INTS: {
    std::vector<Tensor> ints;
    ints.reserve(attrib.ints_size());
    for (const auto &i : attrib.ints()) {
      ints.push_back(Tensor::Scalar(i));
    }
    return std::make_pair(attrib.name(), Tensor::List(std::move(ints)));
  }
  case onnx::AttributeProto_AttributeType_STRINGS: {
    std::vector<Tensor> strings;
    strings.reserve(attrib.strings_size());
    for (const auto &str : attrib.strings()) {
      strings.push_back(Tensor::String(str));
    }
    return std::make_pair(attrib.name(), Tensor::List(std::move(strings)));
  }
  case onnx::AttributeProto_AttributeType_TENSORS: {
    std::vector<Tensor> tensors;
    tensors.reserve(attrib.tensors_size());
    for (const auto &tensor : attrib.tensors()) {
      tensors.push_back(parse_tensor(state, tensor));
    }
    return std::make_pair(attrib.name(), Tensor::List(std::move(tensors)));
  }
  case onnx::AttributeProto_AttributeType_GRAPHS:
    throw std::runtime_error(fmt::format(
        "vkcnn: Node {} attribute {} invalid. List of subgraph attributes "
        "are not supported.",
        nodeName, name));
  case onnx::AttributeProto_AttributeType_SPARSE_TENSORS:
    throw std::runtime_error(fmt::format(
        "vkcnn: Node {} attribute {} invalid. List of sparse tensor "
        "attributes are not supported.",
        nodeName, name));
  case onnx::AttributeProto_AttributeType_TYPE_PROTOS:
    throw std::runtime_error(fmt::format(
        "vkcnn: Node {} attribute {} invalid. List of type prototype "
        "attributes are not supported.",
        nodeName, name));
  case onnx::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MIN_SENTINEL_DO_NOT_USE_:
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} invalid", nodeName, name));
  case onnx::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MAX_SENTINEL_DO_NOT_USE_:
    throw std::runtime_error(
        fmt::format("vkcnn: Node {} attribute {} invalid", nodeName, name));
  }
  throw std::logic_error("unreachable");
}

} // namespace vkcnn::details
