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
  DumpCsv,
  Reweight,
  QueryDeviceInfo,
  MergeDeviceInfo,
  Dump,
};

std::optional<CommandToken> parse_command(std::string_view str);

template <>
struct fmt::formatter<CommandToken> {

  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(CommandToken cmd, FormatContext& ctx) const {
    std::string_view name;

    switch (cmd) {
      case CommandToken::Compile:   name = "compile"; break;
      case CommandToken::Populate:  name = "populate"; break;
      case CommandToken::Bench:     name = "bench"; break;
      case CommandToken::Version:   name = "version"; break;
      case CommandToken::Help:      name = "help"; break;
      case CommandToken::Infer:     name = "infer"; break;
      case CommandToken::DumpCsv:   name = "dumpcsv"; break;
      case CommandToken::Reweight: name = "reweight"; break;
      case CommandToken::QueryDeviceInfo: name = "query-device-info"; break;
      case CommandToken::MergeDeviceInfo: name = "merge-device-info"; break;
      case CommandToken::Dump: name = "dump"; break; 
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};
