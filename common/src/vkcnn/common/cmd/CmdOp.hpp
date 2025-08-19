#pragma once

#include "vkcnn/common/cmd/CmdOpActivation.hpp"
#include "vkcnn/common/cmd/CmdOpBarrier.hpp"
#include "vkcnn/common/cmd/CmdOpConv.hpp"
#include "vkcnn/common/cmd/CmdOpCopy.hpp"
#include "vkcnn/common/cmd/CmdOpPool.hpp"
#include "vkcnn/common/cmd/CmdOpUpsample.hpp"
#include <compare>
#include <cstddef>
#include <array>

namespace vkcnn {

enum class CmdOpTag {
  Conv,
  Activation,
  Copy,
  Pool,
  Upsample,
  Barrier,
  Last = Barrier,
};

class CmdOp;

namespace details {

struct CmdOpLiteral {
  friend CmdOp;

  constexpr auto operator<=>(const CmdOpLiteral &) const = default;

  inline constexpr std::size_t alignment() const {
    return s_alignments[static_cast<std::size_t>(m_tag)];
  }

  inline constexpr std::size_t size() const {
    return s_sizes[static_cast<std::size_t>(m_tag)];
  }

private:
  static constexpr std::size_t Count =
      static_cast<std::size_t>(CmdOpTag::Last) + 1;
  static constexpr std::array<std::size_t, Count> s_alignments{
      alignof(CmdOpConv), alignof(CmdOpActivation), alignof(CmdOpCopy),   //
      alignof(CmdOpPool), alignof(CmdOpUpsample),   alignof(CmdOpBarrier) //
  };

  static constexpr std::array<std::size_t, Count> s_sizes{
      sizeof(CmdOpConv), sizeof(CmdOpActivation), sizeof(CmdOpCopy),   //
      sizeof(CmdOpPool), sizeof(CmdOpUpsample),   sizeof(CmdOpBarrier) //
  };

  explicit constexpr CmdOpLiteral(CmdOpTag tag) : m_tag(tag) {}

  CmdOpTag m_tag;
};

} // namespace details

class CmdOp {

public:
  CmdOp(CmdOpTag tag) : m_literal(tag) {}
  CmdOp(details::CmdOpLiteral literal) : m_literal(literal) {}

  static constexpr details::CmdOpLiteral Conv{CmdOpTag::Conv};
  static constexpr details::CmdOpLiteral Activation{CmdOpTag::Activation};
  static constexpr details::CmdOpLiteral Copy{CmdOpTag::Copy};
  static constexpr details::CmdOpLiteral Pool{CmdOpTag::Pool};
  static constexpr details::CmdOpLiteral Upsample{CmdOpTag::Upsample};
  static constexpr details::CmdOpLiteral Barrier{CmdOpTag::Barrier};

  CmdOpTag tag() const { return m_literal.m_tag; }

  std::size_t alignment() const { return m_literal.alignment(); }
  std::size_t size() const { return m_literal.size(); }

  constexpr auto operator<=>(const CmdOp &) const = default;

  static constexpr inline std::size_t Count = details::CmdOpLiteral::Count;

private:
  details::CmdOpLiteral m_literal;
};

} // namespace vkcnn
