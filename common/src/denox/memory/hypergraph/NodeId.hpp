#pragma once

#include <cstdint>
#include <fmt/format.h>
#include <limits>

namespace denox::memory {

struct NodeId {
public:
  static constexpr std::uint64_t NullId{
      std::numeric_limits<std::uint64_t>::max()};

  explicit constexpr NodeId() : m_id(NullId) {}
  explicit constexpr NodeId(std::uint64_t id) : m_id(id) {}

  constexpr explicit operator std::uint64_t() const { return m_id; }

  constexpr explicit operator bool() const { return m_id != NullId; }

  constexpr std::uint64_t operator*() const { return m_id; }

  friend bool operator==(const NodeId &lhs, const NodeId &rhs) {
    return lhs.m_id == rhs.m_id;
  }

  friend bool operator!=(const NodeId &lhs, const NodeId &rhs) {
    return lhs.m_id != rhs.m_id;
  }

  friend bool operator<(const NodeId& lhs, const NodeId& rhs) {
    return lhs.m_id < rhs.m_id;
  }

  friend bool operator>(const NodeId& lhs, const NodeId& rhs) {
    return lhs.m_id > rhs.m_id;
  }

private:
  std::uint64_t m_id;
};

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::NodeId> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::NodeId &nid, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "<{}>", static_cast<std::uint64_t>(nid));
  }
};
