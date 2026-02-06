#include "denox/cli/parser/parse.hpp"

#include "denox/cli/parser/action.hpp"
#include "denox/cli/parser/lex/lex.hpp"
#include "denox/cli/parser/parse_commands.hpp"
#include <fmt/format.h>
#include <span>

Action parse_argv(int argc, char **argv) {
  std::vector<Token> tokens = lex(argc, argv);

  // fmt::println("\n");
  // for (const auto &token : tokens) {
  //   fmt::println("{}", token);
  // }
  // fmt::println(
  //     "\n\n[kistenklaus@kiste denox]$ denox compile net.onnx -o net.dnx");

  if (tokens.empty()) {
    return HelpAction(HelpScope::Global);
  }
  const Token &first = tokens.front();

  if (first.kind() != TokenKind::Command) {
    switch (first.kind()) {
    case TokenKind::Option:
      throw std::runtime_error("expected command, found option '" +
                               fmt::format("--{}", first.option()));

    case TokenKind::Literal:
      throw std::runtime_error("expected command, found positional argument '" +
                               fmt::format("{}", first.literal().view()));

    case TokenKind::Pipe:
      throw std::runtime_error("expected command, found pipe '-'");
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
  case CommandToken::DumpCsv:
    return parse_dumpcsv(std::span{tokens.begin() + 1, tokens.end()});
  }
  throw std::runtime_error("unreachable");
}
