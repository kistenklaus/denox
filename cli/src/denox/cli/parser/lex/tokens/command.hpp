#pragma once
#include "fmt/core.h"
#include <optional>

enum class CommandToken {
  Compile,
  Populate,
  Bench,
  Infer,
  Version,
  Help,
  DumpCsv
};

std::optional<CommandToken> parse_command(std::string_view str);

template <>
struct fmt::formatter<CommandToken> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(CommandToken cmd, FormatContext &ctx) const {
    std::string_view name;

    switch (cmd) {
    case CommandToken::Compile:
      name = "compile";
      break;
    case CommandToken::Populate:
      name = "populate";
      break;
    case CommandToken::Bench:
      name = "bench";
      break;
    case CommandToken::Version:
      name = "version";
      break;
    case CommandToken::Help:
      name = "help";
      break;
    case CommandToken::Infer:
      name = "infer";
      break;
    case CommandToken::DumpCsv:
      name = "dumpcsv";
      break;
    }

    return fmt::formatter<std::string_view>::format(name, ctx);
  }
};
