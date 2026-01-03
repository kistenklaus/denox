#include "parser/parse_commands.hpp"
#include "alloc/monotone_alloc.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include "parser/errors.hpp"
#include "parser/parse_artefact.hpp"
#include "parser/parse_options.hpp"
#include <fmt/format.h>
#include <stdexcept>
#include <string>

Action parse_compile(std::span<const Token> tokens) {
  if (tokens.empty()) {
    throw ParseError("missing input model");
  }

  // --- Parse input artefact (first positional) ---
  ArtefactParseResult inputRes = parse_artefact(tokens.front());
  if (inputRes.artefact && inputRes.artefact->kind() != ArtefactKind::Onnx) {
    throw ParseError(
        fmt::format("expected ONNX model as first argument, got {}",
                    inputRes.artefact->kind()));
  }

  if (!inputRes.artefact) {
    const Token &tok = tokens.front();

    switch (*inputRes.error) {

    case ArtefactParseError::PathDoesNotExist:
      throw ParseError(
          fmt::format("file not found: '{}'", tok.literal().as_path()));

    case ArtefactParseError::UnrecognizedFormat:
      throw ParseError(
          fmt::format("'{}' is not a valid ONNX model", describe_token(tok)));

    case ArtefactParseError::NotAnArtefactToken:
      throw ParseError(fmt::format(
          "expected ONNX model as first positional argument, got {}",
          describe_token(tok)));
    case ArtefactParseError::DatabasePiped:
      throw ParseError("database input via pipe is not supported");
    case ArtefactParseError::AmbiguousFormat:
      throw ParseError(fmt::format(
          "input '{}' matches multiple artefact formats", describe_token(tok)));

    default:
      throw ParseError(
          fmt::format("invalid input '{}'\n"
                      "usage: denox compile <model.onnx> -o <output.dnx>",
                      describe_token(tok)));
    }
  }

  OnnxArtefact input = inputRes.artefact->onnx();

  // --- Required output ---
  std::optional<IOEndpoint> output;

  // --- Optional ---
  std::optional<DbArtefact> database;
  denox::CompileOptions options{}; // defaults
  // std::memset(&options, 0, sizeof(denox::CompileOptions));

  std::unordered_map<std::string, denox::BufferDescription> inputDescriptions;
  std::unordered_map<std::string, denox::BufferDescription> outputDescriptions;

  std::vector<denox::OptimizationAssumption> assumptions;

  options.srcType = denox::SrcType::Onnx;

  uint32_t i = 1;
  while (i < tokens.size()) {
    uint32_t jump = 0;
    auto tail = std::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(fmt::format("unexpected positional argument {}",
                                   describe_token(tokens[i])));
    }

    // --- options with arguments ---
    if ((jump = parse_output(tail, &output))) {
      i += jump;
      continue;
    }

    if ((jump = parse_database(tail, &database))) {
      i += jump;
      continue;
    }
    //
    if ((jump = parse_input_layout(tail, inputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_output_layout(tail, outputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_input_shape(tail, inputDescriptions))) {
      i += jump;
      continue;
    }
    if ((jump = parse_output_shape(tail, outputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_input_type(tail, inputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_output_type(tail, outputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_input_storage(tail, inputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_output_storage(tail, outputDescriptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_target_env(tail, &options.device.apiVersion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_device(tail, &options.device.deviceName))) {
      i += jump;
      continue;
    }

    // --- boolean flags ---
    if ((jump = parse_spirv_optimize(tail, &options.spirvOptions.optimize))) {
      i += jump;
      continue;
    }

    if ((jump =
             parse_spirv_debug_info(tail, &options.spirvOptions.debugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_coopmat(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_fusion(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_memory_concat(tail,
                                            &options.features.memory_concat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_verbose(tail, &options.verbose))) {
      i += jump;
      continue;
    }

    if ((jump = parse_quiet(tail, &options.quiet))) {
      i += jump;
      continue;
    }

    if ((jump = parse_assume(tail, assumptions))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("unknown option {}", describe_token(tokens[i])));
  }

  if (inputDescriptions.empty()) {
    options.inputDescriptionCount = 0;
    options.inputDescriptions = nullptr;
  } else {
    options.inputDescriptionCount =
        static_cast<uint32_t>(inputDescriptions.size());
    options.inputDescriptions =
        static_cast<denox::BufferDescription *>(mono_alloc(
            sizeof(denox::BufferDescription) * inputDescriptions.size()));
    size_t x = 0;
    for (auto &[name, value] : inputDescriptions) {
      char *cstr = static_cast<char *>(mono_alloc(name.size() + 1));
      std::memset(cstr, 0, name.size() + 1);
      std::memcpy(cstr, name.data(), name.size());
      value.name = cstr;
      options.inputDescriptions[x++] = value;
    }
  }

  if (outputDescriptions.empty()) {
    options.outputDescriptionCount = 0;
    options.outputDescriptions = nullptr;
  } else {
    options.outputDescriptionCount =
        static_cast<uint32_t>(outputDescriptions.size());
    options.outputDescriptions =
        static_cast<denox::BufferDescription *>(mono_alloc(
            sizeof(denox::BufferDescription) * inputDescriptions.size()));
    size_t x = 0;
    for (auto &[name, value] : outputDescriptions) {
      char *cstr = static_cast<char *>(mono_alloc(name.size() + 1));
      std::memset(cstr, 0, name.size() + 1);
      std::memcpy(cstr, name.data(), name.size());
      value.name = cstr;
      options.outputDescriptions[x++] = value;
    }
  }

  if (assumptions.empty()) {
    options.optimizationAssumptionCount = 0;
    options.optimizationAssumptions = nullptr;
  } else {
    options.optimizationAssumptionCount =
        static_cast<uint32_t>(assumptions.size());
    options.optimizationAssumptions =
        static_cast<denox::OptimizationAssumption *>(mono_alloc(
            sizeof(denox::OptimizationAssumption) * assumptions.size()));
    size_t x = 0;
    for (const auto &value : assumptions) {
      options.optimizationAssumptions[x++] = value;
    }
  }

  if (!output) {
    throw ParseError("missing output file");
  }

  return CompileAction{
      .input = input,
      .output = *output,
      .database = database,
      .options = options,
  };
}

Action parse_populate(std::span<const Token> tokens) {
  std::optional<OnnxArtefact> model;
  std::optional<DbArtefact> database;

  // compile options
  denox::CompileOptions options; // pick some defaults.

  // TODO: Parse

  return PopulateAction{
      .model = *model,
      .database = *database,
      .options = options,
  };
}

Action parse_bench(std::span<const Token> tokens) {

  // required
  std::optional<Artefact> target;

  // optional
  std::optional<DbArtefact> database;
  std::optional<denox::CompileOptions> options;
  std::optional<denox::Shape> inputSize;

  // TODO: Parse

  return BenchAction{
      .target = *target,
      .database = database,
      .options = options,
      .inputSize = inputSize,
  };
}

Action parse_infer(std::span<const Token> tokens) {
  // required
  std::optional<Artefact> model;
  std::optional<IOEndpoint> input;
  std::optional<IOEndpoint> output;

  // optional
  std::optional<DbArtefact> db;
  std::optional<denox::CompileOptions> options;

  // TODO: Parse

  return InferAction{
      .model = *model,
      .input = *input,
      .output = *output,
      .database = db,
      .options = options,
  };
}

Action parse_version(std::span<const Token> tokens) {
  return Action::version();
}

Action parse_help(std::span<const Token> tokens) {
  return HelpAction(HelpScope::Global);
}
