#include "parser/parse_commands.hpp"
#include "parser/parse_artefact.hpp"
#include <fmt/base.h>
#include <stdexcept>

Action parse_compile(std::span<const Token> tokens) {
  if (tokens.size() < 1) {
    throw std::runtime_error("expected at least one position argument! pretty error please");
  }

  std::optional<Artefact> inputArtefact = parse_artefact(tokens.front());
  if (!inputArtefact.has_value()) {
    throw std::runtime_error("no artefact");
  }

  if (inputArtefact->kind() != ArtefactKind::Onnx) {
    throw std::runtime_error("artefact must be onnx");
  }

  fmt::println("parsed input");

  OnnxArtefact input = inputArtefact->onnx();





  std::optional<IOEndpoint> output;

  std::optional<DbArtefact> database;
  denox::CompileOptions options; // pick some defaults

  // TODO: Parse

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

