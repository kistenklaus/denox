#include "denox/compiler/frontend/onnx/details/values/Attribute.hpp"
#include "denox/diag/unreachable.hpp"
#include <onnx.pb.h>
#include <stdexcept>

namespace denox::onnx::details {

memory::string_view AttributeKind_name(AttributeKind k) noexcept {
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

Attribute Attribute::Int(std::int64_t v) { return Attribute{Rep{v}}; }

Attribute Attribute::Float(float v) { return Attribute{Rep{v}}; }

Attribute Attribute::String(memory::string v) {
  return Attribute{Rep{std::move(v)}};
}

Attribute Attribute::Tensor(const HostTensor &t) { return Attribute{Rep{t}}; }

Attribute Attribute::Tensor(HostTensor &&t) {
  return Attribute{Rep{std::move(t)}};
}

Attribute Attribute::Graph(GraphAttr g) { return Attribute{Rep{std::move(g)}}; }

Attribute Attribute::Ints(memory::vector<std::int64_t> v) {
  return Attribute{Rep{std::move(v)}};
}

Attribute Attribute::Floats(memory::vector<float> v) {
  return Attribute{Rep{std::move(v)}};
}

Attribute Attribute::Strings(memory::vector<memory::string> v) {
  return Attribute{Rep{std::move(v)}};
}

Attribute Attribute::Tensors(memory::vector<HostTensor> v) {
  return Attribute{Rep{std::move(v)}};
}

Attribute Attribute::Graphs(memory::vector<GraphAttr> v) {
  return Attribute{Rep{std::move(v)}};
}

AttributeKind Attribute::kind() const {
  if (std::holds_alternative<std::int64_t>(m_rep))
    return AttributeKind::Int;
  if (std::holds_alternative<float>(m_rep))
    return AttributeKind::Float;
  if (std::holds_alternative<memory::string>(m_rep))
    return AttributeKind::String;
  if (std::holds_alternative<HostTensor>(m_rep))
    return AttributeKind::Tensor;
  if (std::holds_alternative<GraphAttr>(m_rep))
    return AttributeKind::Graph;
  if (std::holds_alternative<memory::vector<float>>(m_rep))
    return AttributeKind::Floats;
  if (std::holds_alternative<memory::vector<std::int64_t>>(m_rep))
    return AttributeKind::Ints;
  if (std::holds_alternative<memory::vector<memory::string>>(m_rep))
    return AttributeKind::Strings;
  if (std::holds_alternative<memory::vector<HostTensor>>(m_rep))
    return AttributeKind::Tensors;
  if (std::holds_alternative<memory::vector<GraphAttr>>(m_rep))
    return AttributeKind::Graphs;
  assert(false);
  return AttributeKind::Ints;
}

bool Attribute::isInt() const {
  return std::holds_alternative<std::int64_t>(m_rep);
}

bool Attribute::isFloat() const { return std::holds_alternative<float>(m_rep); }

bool Attribute::isString() const {
  return std::holds_alternative<memory::string>(m_rep);
}

bool Attribute::isTensor() const {
  return std::holds_alternative<HostTensor>(m_rep);
}

bool Attribute::isGraph() const {
  return std::holds_alternative<GraphAttr>(m_rep);
}

bool Attribute::isInts() const {
  return std::holds_alternative<memory::vector<std::int64_t>>(m_rep);
}

bool Attribute::isFloats() const {
  return std::holds_alternative<memory::vector<float>>(m_rep);
}

bool Attribute::isStrings() const {
  return std::holds_alternative<memory::vector<memory::string>>(m_rep);
}

bool Attribute::isTensors() const {
  return std::holds_alternative<memory::vector<HostTensor>>(m_rep);
}

bool Attribute::isGraphs() const {
  return std::holds_alternative<memory::vector<GraphAttr>>(m_rep);
}

std::int64_t Attribute::i() const {
  ensure(AttributeKind::Int);
  return std::get<std::int64_t>(m_rep);
}

float Attribute::f() const {
  ensure(AttributeKind::Float);
  return std::get<float>(m_rep);
}

const memory::string &Attribute::s() const {
  ensure(AttributeKind::String);
  return std::get<memory::string>(m_rep);
}

const HostTensor &Attribute::t() const {
  ensure(AttributeKind::Tensor);
  return std::get<HostTensor>(m_rep);
}

const GraphAttr &Attribute::g() const {
  ensure(AttributeKind::Graph);
  return std::get<GraphAttr>(m_rep);
}

const memory::vector<std::int64_t> &Attribute::ints() const {
  ensure(AttributeKind::Ints);
  return std::get<memory::vector<std::int64_t>>(m_rep);
}

const memory::vector<float> &Attribute::floats() const {
  ensure(AttributeKind::Floats);
  return std::get<memory::vector<float>>(m_rep);
}

const memory::vector<memory::string> &Attribute::strings() const {
  ensure(AttributeKind::Strings);
  return std::get<memory::vector<memory::string>>(m_rep);
}

const memory::vector<HostTensor> &Attribute::tensors() const {
  ensure(AttributeKind::Tensors);
  return std::get<memory::vector<HostTensor>>(m_rep);
}

const memory::vector<GraphAttr> &Attribute::graphs() const {
  ensure(AttributeKind::Graphs);
  return std::get<memory::vector<GraphAttr>>(m_rep);
}

memory::string_view Attribute::kindName() const noexcept {
  return AttributeKind_name(kind());
}

Attribute::Attribute(Rep rep) : m_rep(std::move(rep)) {}
void Attribute::ensure(AttributeKind k) const {
  if (kind() != k)
    throw std::runtime_error(
        fmt::format("Attribute: wrong kind. Expected {}, Got {}",
                    AttributeKind_name(k), AttributeKind_name(kind())));
}

static ::onnx::AttributeProto_AttributeType
infer_attribute_type_loose(const ::onnx::AttributeProto &a,
                           memory::string_view nodeName) {
  using AT = ::onnx::AttributeProto_AttributeType;

  const memory::string aname =
      a.name().empty() ? memory::string("<unnamed>") : a.name();

  // Not supported until you implement function-attribute indirection.
  if (!a.ref_attr_name().empty()) {
    throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
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
        throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=TENSOR but no tensor payload");
      break;
    case AT::AttributeProto_AttributeType_SPARSE_TENSOR:
      if (!a.has_sparse_tensor())
        throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=SPARSE_TENSOR but no payload");
      break;
    case AT::AttributeProto_AttributeType_GRAPH:
      if (!a.has_g())
        throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                                 "\" attribute \"" + aname +
                                 "\": type=GRAPH but no graph payload");
      break;
    case AT::AttributeProto_AttributeType_TYPE_PROTO:
      if (!a.has_tp())
        throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
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
      break;
    case AT::
        AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MIN_SENTINEL_DO_NOT_USE_:
    case AT::
        AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MAX_SENTINEL_DO_NOT_USE_:
      diag::unreachable();
    }
    return explicit_t;
  }

  // ---------- Implicit inference path ----------
  auto pick_only = [&](bool cond, AT t, AT &acc, bool &found) {
    if (!cond)
      return;
    if (found && acc != t) {
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
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
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
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
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred TENSOR but no tensor payload");
    break;
  case AT::AttributeProto_AttributeType_GRAPH:
    if (!a.has_g())
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred GRAPH but no graph payload");
    break;
  case AT::AttributeProto_AttributeType_SPARSE_TENSOR:
    if (!a.has_sparse_tensor())
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
                               "\" attribute \"" + aname +
                               "\": inferred SPARSE_TENSOR but no payload");
    break;
  case AT::AttributeProto_AttributeType_TYPE_PROTO:
    if (!a.has_tp())
      throw std::runtime_error("vkcnn: node \"" + memory::string(nodeName) +
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
    break; // failed to resolve type!
  case AT::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MAX_SENTINEL_DO_NOT_USE_:
  case AT::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MIN_SENTINEL_DO_NOT_USE_:
    diag::unreachable();
  }
  return ret;
}

NamedAttribute Attribute::parse(const ::onnx::AttributeProto &a,
                                const io::Path &externalDir,
                                memory::string_view nodeName) {

  const memory::string name =
      a.name().empty() ? memory::string("<unnamed>") : a.name();

  if (!a.ref_attr_name().empty()) {
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": ref_attr_name is unsupported",
        nodeName, name));
  }

  using AT = ::onnx::AttributeProto_AttributeType;
  const AT at = infer_attribute_type_loose(a, nodeName);

  switch (at) {
  case AT::AttributeProto_AttributeType_UNDEFINED:
    // Ambiguous (e.g., scalar default 0 or empty list w/o explicit type).
    // Safer to fail loudly than guess.
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": unable to infer attribute type "
        "(ambiguous/absent payload). Exporter should set explicit type.",
        nodeName, name));

  case AT::AttributeProto_AttributeType_FLOAT:
    return {a.name(), Attribute::Float(a.f())};

  case AT::AttributeProto_AttributeType_INT:
    return {a.name(), Attribute::Int(a.i())};

  case AT::AttributeProto_AttributeType_STRING:
    return {a.name(), Attribute::String(a.s())};

  case AT::AttributeProto_AttributeType_TENSOR: {
    // ONNX attribute tensors are compile-time constants → HostTensor
    HostTensor ht = HostTensor::parse(a.t(), externalDir);
    return {a.name(), Attribute::Tensor(std::move(ht))};
  }

  case AT::AttributeProto_AttributeType_GRAPH:
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": GRAPH attributes unsupported",
        nodeName, name));

  case AT::AttributeProto_AttributeType_SPARSE_TENSOR:
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": SPARSE_TENSOR unsupported",
        nodeName, name));

  case AT::AttributeProto_AttributeType_TYPE_PROTO:
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": TYPE_PROTO unsupported", nodeName,
        name));

  case AT::AttributeProto_AttributeType_FLOATS: {
    memory::vector<float> v;
    v.reserve(static_cast<std::size_t>(a.floats_size()));
    for (float f : a.floats())
      v.push_back(f);
    return {a.name(), Attribute::Floats(std::move(v))};
  }

  case AT::AttributeProto_AttributeType_INTS: {
    memory::vector<std::int64_t> v;
    v.reserve(static_cast<std::size_t>(a.ints_size()));
    for (auto vi : a.ints())
      v.push_back(static_cast<std::int64_t>(vi));
    return {a.name(), Attribute::Ints(std::move(v))};
  }

  case AT::AttributeProto_AttributeType_STRINGS: {
    memory::vector<memory::string> v;
    v.reserve(static_cast<std::size_t>(a.strings_size()));
    for (const auto &s : a.strings())
      v.emplace_back(s);
    return {a.name(), Attribute::Strings(std::move(v))};
  }

  case AT::AttributeProto_AttributeType_TENSORS: {
    memory::vector<HostTensor> v;
    v.reserve(static_cast<std::size_t>(a.tensors_size()));
    for (const auto &tp : a.tensors()) {
      v.emplace_back(HostTensor::parse(tp, externalDir));
    }
    return {a.name(), Attribute::Tensors(std::move(v))};
  }

  case AT::AttributeProto_AttributeType_GRAPHS:
    throw std::runtime_error(
        fmt::format("vkcnn: node \"{}\" attribute \"{}\": GRAPHS unsupported",
                    nodeName, name));

  case AT::AttributeProto_AttributeType_SPARSE_TENSORS:
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": SPARSE_TENSORS unsupported",
        nodeName, name));

  case AT::AttributeProto_AttributeType_TYPE_PROTOS:
    throw std::runtime_error(fmt::format(
        "vkcnn: node \"{}\" attribute \"{}\": TYPE_PROTOS unsupported",
        nodeName, name));
  case AT::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MAX_SENTINEL_DO_NOT_USE_:
  case AT::
      AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MIN_SENTINEL_DO_NOT_USE_:
    diag::unreachable();

  default:
    break;
  }

  throw std::runtime_error(fmt::format(
      "vkcnn: node \"{}\" attribute \"{}\": unsupported/unknown attribute type",
      nodeName, name));
}

} // namespace denox::onnx::details
