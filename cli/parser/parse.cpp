#include "parser/parse.hpp"

#include "denox/compiler.hpp"
#include "parser/action.hpp"
#include "parser/lex/lex.hpp"
#include "parser/parse_commands.hpp"
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <span>



Action parse_argv(int argc, char **argv) {
  std::vector<Token> tokens = lex(argc, argv);

  for (const auto &token : tokens) {
    fmt::println("{}", token);
  }

  if (tokens.empty()) {
    return HelpAction(HelpScope::Global);
  }
  const Token &first = tokens.front();

  if (first.kind() != TokenKind::Command) {
    switch (first.kind()) {
    case TokenKind::Option:
      throw std::runtime_error(
          "expected command, found option '" +
          fmt::format("--{}", first.option()) +
          "'\n"
          "hint: commands must come first (e.g. 'denox compile ...')");

    case TokenKind::Literal:
      throw std::runtime_error("expected command, found positional argument '" +
                               fmt::format("{}", first.literal().view()) +
                               "'\n"
                               "hint: denox requires an explicit command "
                               "(e.g. 'denox compile', 'denox infer')");

    case TokenKind::Pipe:
      throw std::runtime_error(
          "expected command, found pipe '-'\n"
          "hint: pipes may only appear as inputs or outputs, "
          "not as commands");

    case TokenKind::Command:
      throw std::runtime_error("invalid state");

    default:
      throw std::runtime_error("unexpected token");
    }
  }

  const auto &cmd = first.command();
  switch (cmd) {
  case CommandToken::Compile:
    return parse_compile(std::span{tokens.begin() + 1, tokens.end()});
  case CommandToken::Populate:
    return parse_populate(std::span{tokens.begin() + 1, tokens.end()});
  case CommandToken::Bench:
    return parse_bench(std::span{tokens.begin() + 1, tokens.end()});
  case CommandToken::Infer:
    return parse_infer(std::span{tokens.begin() + 1, tokens.end()});
  case CommandToken::Version:
    return parse_version(std::span{tokens.begin() + 1, tokens.end()});
  case CommandToken::Help:
    return parse_help(std::span{tokens.begin() + 1, tokens.end()});
  }
  throw std::runtime_error("unreachable");
}
