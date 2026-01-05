#include "denox/cli/parser/errors.hpp"

std::string describe_token(const Token &t) {

  switch (t.kind()) {
  case TokenKind::Literal:
    return fmt::format("{}", t.literal().view());
  case TokenKind::Pipe:
    return "stdin ('-')";
  case TokenKind::Option:
    return fmt::format("--{}", t.option());
  case TokenKind::Command:
    return fmt::format("{}", t.command());
  }
  return "unknown token";
}
