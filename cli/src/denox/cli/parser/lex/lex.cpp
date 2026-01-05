#include "denox/cli/parser/lex/lex.hpp"

#include "denox/cli/io/Pipe.hpp"
#include "tokens/command.hpp"
#include "tokens/literal.hpp"
#include "tokens/option.hpp"
#include "tokens/token.hpp"

#include <fmt/core.h>
#include <stdexcept>
#include <string_view>
#include <vector>

std::vector<Token> lex(int argc, char **argv) {
  std::vector<Token> tokens;
  tokens.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);

  bool end_of_options = false;

  auto emit_literal_split = [&](std::string_view sv) {
    while (!sv.empty()) {
      auto pos = sv.find_first_of(":=");
      if (pos == std::string_view::npos) {
        tokens.emplace_back(LiteralToken{std::string(sv)});
        break;
      }

      if (pos > 0) {
        tokens.emplace_back(LiteralToken{std::string(sv.substr(0, pos))});
      }

      sv.remove_prefix(pos + 1);
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};

    // 0. end-of-options marker
    if (!end_of_options && arg == "--") {
      end_of_options = true;
      continue;
    }

    // 1. command (even if it looks like an option)
    if (!end_of_options) {
      if (auto cmd = parse_command(arg)) {
        tokens.emplace_back(*cmd);
        continue;
      }
    }

    // 2. stdin/stdout pipe
    if (arg == "-") {
      tokens.emplace_back(Pipe{});
      continue;
    }

    // 3. long option: --foo or --foo=bar
    if (!end_of_options && arg.starts_with("--")) {
      auto body = arg.substr(2);
      auto eq = body.find('=');

      std::string_view name =
          (eq == std::string_view::npos) ? body : body.substr(0, eq);

      if (auto opt = parse_option(name)) {
        tokens.emplace_back(*opt);
      } else {
        throw std::runtime_error(fmt::format("unknown option '--{}'", name));
      }

      if (eq != std::string_view::npos) {
        emit_literal_split(body.substr(eq + 1));
      }

      continue;
    }

    // 4. short options: -o, -vq, etc. (excluding "-")
    if (!end_of_options && arg.starts_with("-")) {
      if (arg.size() == 1) {
        throw std::runtime_error("invalid '-' token");
      }

      for (size_t j = 1; j < arg.size(); ++j) {
        std::string_view key{&arg[j], 1};

        // short command (-h)
        if (auto cmd = parse_command(key)) {
          tokens.emplace_back(*cmd);
        }
        // short option
        else if (auto opt = parse_option(key)) {
          tokens.emplace_back(*opt);
        } else {
          throw std::runtime_error(
              fmt::format("unknown short option '-{}'", key));
        }
      }

      continue;
    }

    // 5. literal (always split on ':' and '=')
    emit_literal_split(arg);
  }

  return tokens;
}
