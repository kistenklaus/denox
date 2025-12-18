#pragma once
#include <cassert>
#include <cstdlib>
#include <fmt/core.h>
#include <variant>

#include "command.hpp"
#include "io/Pipe.hpp"
#include "literal.hpp"
#include "option.hpp"

enum class TokenKind {
  Command,
  Option,
  Pipe,
  Literal,
};

class Token {
public:
  explicit Token(CommandToken cmd) noexcept : m_value(cmd) {}
  explicit Token(OptionToken opt) noexcept : m_value(opt) {}
  explicit Token(Pipe pipe) noexcept : m_value(pipe) {}
  explicit Token(LiteralToken lit) noexcept : m_value(std::move(lit)) {}

  TokenKind kind() const noexcept {
    if (std::holds_alternative<CommandToken>(m_value))
      return TokenKind::Command;
    if (std::holds_alternative<OptionToken>(m_value))
      return TokenKind::Option;
    if (std::holds_alternative<Pipe>(m_value))
      return TokenKind::Pipe;
    if (std::holds_alternative<LiteralToken>(m_value))
      return TokenKind::Literal;

    std::abort();
  }

  CommandToken command() const noexcept {
    assert(kind() == TokenKind::Command);
    return std::get<CommandToken>(m_value);
  }

  OptionToken option() const noexcept {
    assert(kind() == TokenKind::Option);
    return std::get<OptionToken>(m_value);
  }

  Pipe pipe() const noexcept {
    assert(kind() == TokenKind::Pipe);
    return std::get<Pipe>(m_value);
  }

  const LiteralToken &literal() const noexcept {
    assert(kind() == TokenKind::Literal);
    return std::get<LiteralToken>(m_value);
  }

private:
  std::variant<CommandToken, OptionToken, Pipe, LiteralToken> m_value;
};

template <> struct fmt::formatter<Token> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const Token &tok, FormatContext &ctx) const {
    switch (tok.kind()) {
    case TokenKind::Command:
      return fmt::format_to(ctx.out(), "Command({})", tok.command());

    case TokenKind::Option:
      return fmt::format_to(ctx.out(), "Option({})", tok.option());

    case TokenKind::Pipe:
      return fmt::format_to(ctx.out(), "Pipe({})", tok.pipe());

    case TokenKind::Literal:
      return fmt::format_to(ctx.out(), "Literal({})", tok.literal());
    }

    std::abort(); // unreachable
  }
};
