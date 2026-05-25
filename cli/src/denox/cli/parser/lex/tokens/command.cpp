#include "denox/cli/parser/lex/tokens/command.hpp"

std::optional<CommandToken> parse_command(std::string_view str) {
  if (str.empty()) {
    return std::nullopt;
  }
  if (str == "infer") {
    return CommandToken::Infer;
  }

  if (str == "compile") {
    return CommandToken::Compile;
  }
  if (str == "populate") {
    return CommandToken::Populate;
  }
  if (str == "bench") {
    return CommandToken::Bench;
  }
  if (str == "--version") {
    return CommandToken::Version;
  }
  if (str == "--help" || str == "-h") {
    return CommandToken::Help;
  }
  if (str == "dumpcsv") {
    return CommandToken::DumpCsv;
  }
  if (str == "reweight") {
    return CommandToken::Reweight;
  }
  if (str == "query-device-info" || str == "create-device-info") {
    return CommandToken::QueryDeviceInfo;
  }
  if (str == "merge-device-info" || str == "min-device-info") {
    return CommandToken::MergeDeviceInfo;
  }

  return std::nullopt;
}
