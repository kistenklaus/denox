#include "denox/cli/parser/parse_commands.hpp"
#include "denox/cli/alloc/monotone_alloc.hpp"
#include "denox/cli/io/Pipe.hpp"
#include "denox/cli/parser/action.hpp"
#include "denox/cli/parser/errors.hpp"
#include "denox/cli/parser/parse_artefact.hpp"
#include "denox/cli/parser/parse_options.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
// #include "denox/cli/parser/parse_options.hpp"
#include <absl/strings/str_format.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

Action parse_compile(std::span<const Token> tokens) {
  if (tokens.empty()) {
    throw ParseError("missing input model");
  }

  if (parse_help(tokens, nullptr)) {
    return HelpAction(HelpScope::Compile);
  }

  // --- Parse input artefact (first positional) ---
  ArtefactParseResult inputRes = parse_artefact(tokens.front());
  if (inputRes.artefact && inputRes.artefact->kind() != ArtefactKind::Onnx) {
    throw ParseError(
        fmt::format("expected ONNX model as first argument, got {}",
                    inputRes.artefact->kind()));
  }
  if (inputRes.error.has_value()) {
    switch (*inputRes.error) {
    case ArtefactParseError::NotAnArtefactToken:
      throw ParseError(fmt::format("expected ONNX model as first argument"));
    case ArtefactParseError::PathDoesNotExist:
      throw ParseError(fmt::format("Path does not exist"));
    case ArtefactParseError::UnrecognizedFormat:
      throw ParseError(fmt::format("Unregonized format"));
    case ArtefactParseError::DatabasePiped:
      throw ParseError(fmt::format("Databases cannot be piped"));
    case ArtefactParseError::AmbiguousFormat:
      throw ParseError(fmt::format("Input format is ambiguous"));
    }
  }
  OnnxArtefact input = inputRes.artefact->onnx();
  denox::memory::optional<IOEndpoint> output;
  denox::memory::optional<DbArtefact> database;
  denox::compiler::CompileOptions options{};
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion = denox::ApiVersion::VULKAN_1_4;

  bool spirv_nonSemanticDebugInfo = false;
  bool spirv_debugInfo = false;

  denox::memory::hash_map<denox::memory::string,
                          denox::compiler::InterfaceTensorDescriptor>
      tensorDescriptors;
  denox::memory::hash_map<denox::memory::string, int64_t> assumptions;
  bool help = false;

  bool fusion = true;

  // parse remaining arguments
  uint32_t i = 1;
  while (i < tokens.size()) {
    uint32_t jump = 0;

    auto tail = denox::memory::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(fmt::format("unexpected positional argument {}",
                                   describe_token(tokens[i])));
    }

    if ((jump = parse_help(tokens, &help))) {
      i += jump;
      continue;
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

    if ((jump = parse_target_env(tail, &apiVersion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_device(tail, &deviceName))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_optimize(tail, &options.spirv.optimize))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_non_semantic_debug_info(
             tail, &spirv_nonSemanticDebugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_debug_info(tail, &spirv_debugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_coopmat(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_fusion(tail, &fusion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_memory_concat(
             tail, &options.features.enableImplicitConcat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_shape(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_storage(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_format(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_type(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_use_descriptor_sets(tail, &options.descriptorPolicies))) {
      i += jump;
      continue;
    }

    if ((jump = parse_assume(tail, assumptions))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
  }

  options.features.enableConvReluFusion = fusion;
  options.features.enableConcatConvFusion = fusion;

  if (help) {
    return HelpAction(HelpScope::Compile);
  }

  if (spirv_nonSemanticDebugInfo) {
    options.spirv.debugInfo =
        denox::spirv::SpirvDebugInfoLevel::ForceNonSemanticDebugInfo;
  } else if (spirv_debugInfo) {
    options.spirv.debugInfo = denox::spirv::SpirvDebugInfoLevel::Enable;
  }

  for (auto &[name, interface] : tensorDescriptors) {
    interface.name = name;
    options.interfaceDescriptors.push_back(interface);
  }

  for (auto &[name, value] : assumptions) {
    options.assumptions.valueAssumptions.emplace_back(name, value);
  }

  IOEndpoint out = output.value_or([&]() -> IOEndpoint {
    switch (input.endpoint.kind()) {
    case IOEndpointKind::Path:
      return input.endpoint.path().with_extension("dnx");
    case IOEndpointKind::Pipe:
      return denox::io::Path::assets() / "a.dnx";
    }
    denox::diag::unreachable();
  }());

  return CompileAction{
      .input = std::move(input),
      .output = out,
      .deviceName = deviceName,
      .apiVersion = apiVersion,
      .database = std::move(database),
      .options = options,
  };
}

Action parse_populate(std::span<const Token> tokens) {
  if (tokens.empty()) {
    throw ParseError("missing database and model");
  }

  if (parse_help(tokens, nullptr)) {
    return HelpAction(HelpScope::Populate);
  }

  if (tokens.size() < 2) {
    throw ParseError(fmt::format("populate expectes two positional arguments"));
  }

  const auto &dbToken = tokens.front();
  if (dbToken.kind() != TokenKind::Literal) {
    throw ParseError(fmt::format(
        "expected database path as first argument, got {}", dbToken));
  }
  if (!dbToken.literal().is_path()) {
    throw ParseError(
        fmt::format("expected database path as first argument, got {}",
                    dbToken.literal().view()));
  }
  DbArtefact database(dbToken.literal().as_path());

  // --- Parse input artefact (first positional) ---
  ArtefactParseResult inputRes = parse_artefact(tokens[1]);
  if (inputRes.artefact && inputRes.artefact->kind() != ArtefactKind::Onnx) {
    throw ParseError(
        fmt::format("expected ONNX model as second argument, got {}",
                    inputRes.artefact->kind()));
  }
  if (inputRes.error.has_value()) {
    switch (*inputRes.error) {
    case ArtefactParseError::NotAnArtefactToken:
      denox::diag::invalid_state();
    case ArtefactParseError::PathDoesNotExist:
      throw ParseError(fmt::format("Path does not exist"));
    case ArtefactParseError::UnrecognizedFormat:
      throw ParseError(fmt::format("Unregonized format"));
    case ArtefactParseError::DatabasePiped:
      throw ParseError(fmt::format("Databases cannot be piped"));
    case ArtefactParseError::AmbiguousFormat:
      throw ParseError(fmt::format("Input format is ambiguous"));
    }
  }
  OnnxArtefact model = inputRes.artefact->onnx();
  denox::compiler::CompileOptions options{};
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion = denox::ApiVersion::VULKAN_1_4;

  bool spirv_nonSemanticDebugInfo = false;
  bool spirv_debugInfo = false;

  denox::memory::hash_map<denox::memory::string,
                          denox::compiler::InterfaceTensorDescriptor>
      tensorDescriptors;
  denox::memory::hash_map<denox::memory::string, int64_t> assumptions;

  bool help = false;
  bool fusion = true;

  // parse remaining arguments
  uint32_t i = 2;
  while (i < tokens.size()) {
    uint32_t jump = 0;

    auto tail = denox::memory::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(fmt::format("unexpected positional argument {}",
                                   describe_token(tokens[i])));
    }

    if ((jump = parse_help(tail, &help))) {
      i += jump;
      continue;
    }

    if ((jump = parse_target_env(tail, &apiVersion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_device(tail, &deviceName))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_optimize(tail, &options.spirv.optimize))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_non_semantic_debug_info(
             tail, &spirv_nonSemanticDebugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_debug_info(tail, &spirv_debugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_coopmat(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_fusion(tail, &fusion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_memory_concat(
             tail, &options.features.enableImplicitConcat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_shape(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_storage(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_format(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_type(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_use_descriptor_sets(tail, &options.descriptorPolicies))) {
      i += jump;
      continue;
    }

    if ((jump = parse_assume(tail, assumptions))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
  }

  if (help) {
    return HelpAction(HelpScope::Populate);
  }

  options.features.enableConvReluFusion = fusion;
  options.features.enableConcatConvFusion = fusion;

  if (spirv_nonSemanticDebugInfo) {
    options.spirv.debugInfo =
        denox::spirv::SpirvDebugInfoLevel::ForceNonSemanticDebugInfo;
  } else if (spirv_debugInfo) {
    options.spirv.debugInfo = denox::spirv::SpirvDebugInfoLevel::Enable;
  }

  for (auto &[name, interface] : tensorDescriptors) {
    interface.name = name;
    options.interfaceDescriptors.push_back(interface);
  }

  for (auto &[name, value] : assumptions) {
    options.assumptions.valueAssumptions.emplace_back(name, value);
  }

  return PopulateAction{
      .model = std::move(model),
      .database = std::move(database),
      .deviceName = deviceName,
      .apiVersion = apiVersion,
      .options = options,
  };
}

Action parse_bench(std::span<const Token> tokens) {
  if (tokens.empty()) {
    throw ParseError("missing input model");
  }

  if (parse_help(tokens, nullptr)) {
    return HelpAction(HelpScope::Bench);
  }

  // --- Parse input artefact (first positional) ---
  ArtefactParseResult inputRes = parse_artefact(tokens.front());
  if (inputRes.error.has_value()) {
    switch (*inputRes.error) {
    case ArtefactParseError::NotAnArtefactToken:
      denox::diag::invalid_state();
    case ArtefactParseError::PathDoesNotExist:
      throw ParseError(fmt::format("Path does not exist"));
    case ArtefactParseError::UnrecognizedFormat:
      throw ParseError(fmt::format("Unregonized format"));
    case ArtefactParseError::DatabasePiped:
      throw ParseError(fmt::format("Databases cannot be piped"));
    case ArtefactParseError::AmbiguousFormat:
      throw ParseError(fmt::format("Input format is ambiguous"));
    }
  }
  Artefact model = *inputRes.artefact;
  denox::memory::optional<IOEndpoint> output;
  denox::memory::optional<DbArtefact> database;
  denox::compiler::CompileOptions options{};
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion = denox::ApiVersion::VULKAN_1_4;

  denox::runtime::DbBenchOptions dbBenchOptions;

  bool spirv_nonSemanticDebugInfo = false;
  bool spirv_debugInfo = false;

  denox::memory::hash_map<denox::memory::string,
                          denox::compiler::InterfaceTensorDescriptor>
      tensorDescriptors;
  denox::memory::hash_map<denox::memory::string, int64_t> assumptions;

  denox::memory::hash_map<denox::memory::string, int64_t> valueSpecMap;

  bool help = false;
  bool fusion = true;

  // parse remaining arguments
  uint32_t i = 1;
  while (i < tokens.size()) {
    uint32_t jump = 0;

    auto tail = denox::memory::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(fmt::format("unexpected positional argument {}",
                                   describe_token(tokens[i])));
    }

    if ((jump = parse_help(tokens, &help))) {
      i += jump;
      continue;
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

    if ((jump = parse_target_env(tail, &apiVersion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_device(tail, &deviceName))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_optimize(tail, &options.spirv.optimize))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_non_semantic_debug_info(
             tail, &spirv_nonSemanticDebugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_debug_info(tail, &spirv_debugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_coopmat(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_fusion(tail, &fusion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_memory_concat(
             tail, &options.features.enableImplicitConcat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_shape(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_storage(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_format(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_type(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_use_descriptor_sets(tail, &options.descriptorPolicies))) {
      i += jump;
      continue;
    }

    if ((jump = parse_assume(tail, assumptions))) {
      i += jump;
      continue;
    }

    if ((jump = parse_specialize(tail, valueSpecMap))) {
      i += jump;
      continue;
    }

    if ((jump = parse_samples(tail, &dbBenchOptions.minSamples))) {
      i += jump;
      continue;
    }

    if ((jump = parse_relative_error(tail, &dbBenchOptions.maxRelativeError))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
  }

  if (help) {
    return HelpAction(HelpScope::Bench);
  }

  options.features.enableConvReluFusion = fusion;
  options.features.enableConcatConvFusion = fusion;

  if (spirv_nonSemanticDebugInfo) {
    options.spirv.debugInfo =
        denox::spirv::SpirvDebugInfoLevel::ForceNonSemanticDebugInfo;
  } else if (spirv_debugInfo) {
    options.spirv.debugInfo = denox::spirv::SpirvDebugInfoLevel::Enable;
  }

  for (auto &[name, interface] : tensorDescriptors) {
    interface.name = name;
    options.interfaceDescriptors.push_back(interface);
  }

  for (auto &[name, value] : assumptions) {
    options.assumptions.valueAssumptions.emplace_back(name, value);
  }

  denox::memory::vector<denox::ValueSpec> valueSpecs;
  for (auto &[name, value] : valueSpecMap) {
    valueSpecs.emplace_back(name, value);
  }

  return BenchAction{
      .target = std::move(model),

      .deviceName = deviceName,
      .apiVersion = apiVersion,

      .benchOptions = dbBenchOptions,

      .database = std::move(database),
      .options = options,

      .valueSpecs = std::move(valueSpecs),
  };
}

Action parse_infer(std::span<const Token> tokens) {
  if (tokens.empty()) {
    throw ParseError("missing input model");
  }

  if (parse_help(tokens, nullptr)) {
    return HelpAction(HelpScope::Infer);
  }

  // --- Parse input artefact (first positional) ---
  ArtefactParseResult inputRes = parse_artefact(tokens.front());
  if (inputRes.artefact &&
      inputRes.artefact->kind() == ArtefactKind::Database) {
    throw ParseError(fmt::format("expected model as first argument, got {}",
                                 inputRes.artefact->kind()));
  }
  if (inputRes.error.has_value()) {
    switch (*inputRes.error) {
    case ArtefactParseError::NotAnArtefactToken:
      denox::diag::invalid_argument();
    case ArtefactParseError::PathDoesNotExist:
      throw ParseError(fmt::format("Path does not exist"));
    case ArtefactParseError::UnrecognizedFormat:
      throw ParseError(fmt::format("Unregonized format"));
    case ArtefactParseError::DatabasePiped:
      throw ParseError(fmt::format("Databases cannot be piped"));
    case ArtefactParseError::AmbiguousFormat:
      throw ParseError(fmt::format("Input format is ambiguous"));
    }
  }
  Artefact model = *inputRes.artefact;
  denox::memory::optional<DbArtefact> database;
  denox::compiler::CompileOptions options{};
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion = denox::ApiVersion::VULKAN_1_4;

  bool spirv_nonSemanticDebugInfo = false;
  bool spirv_debugInfo = false;

  denox::memory::hash_map<denox::memory::string,
                          denox::compiler::InterfaceTensorDescriptor>
      tensorDescriptors;
  denox::memory::hash_map<denox::memory::string, int64_t> assumptions;

  denox::memory::optional<IOEndpoint> input;
  denox::memory::optional<IOEndpoint> output;


  bool help = false;
  bool fusion = true;

  // parse remaining arguments
  uint32_t i = 1;
  while (i < tokens.size()) {
    uint32_t jump = 0;

    auto tail = denox::memory::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(fmt::format("unexpected positional argument {}",
                                   describe_token(tokens[i])));
    }

    if ((jump = parse_help(tokens, &help))) {
      i += jump;
      continue;
    }

    // --- options with arguments ---
    if ((jump = parse_output(tail, &output))) {
      i += jump;
      continue;
    }

    if ((jump = parse_input(tail, &input))) {
      i += jump;
      continue;
    }

    if ((jump = parse_database(tail, &database))) {
      i += jump;
      continue;
    }

    if ((jump = parse_target_env(tail, &apiVersion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_device(tail, &deviceName))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_optimize(tail, &options.spirv.optimize))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_non_semantic_debug_info(
             tail, &spirv_nonSemanticDebugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_spirv_debug_info(tail, &spirv_debugInfo))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_coopmat(tail, &options.features.coopmat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_fusion(tail, &fusion))) {
      i += jump;
      continue;
    }

    if ((jump = parse_feature_memory_concat(
             tail, &options.features.enableImplicitConcat))) {
      i += jump;
      continue;
    }

    if ((jump = parse_shape(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_storage(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_format(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_type(tail, tensorDescriptors))) {
      i += jump;
      continue;
    }

    if ((jump = parse_use_descriptor_sets(tail, &options.descriptorPolicies))) {
      i += jump;
      continue;
    }

    if ((jump = parse_assume(tail, assumptions))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
  }


  if (help) {
    return HelpAction(HelpScope::Infer);
  }

  options.features.enableConvReluFusion = fusion;
  options.features.enableConcatConvFusion = fusion;

  if (spirv_nonSemanticDebugInfo) {
    options.spirv.debugInfo =
        denox::spirv::SpirvDebugInfoLevel::ForceNonSemanticDebugInfo;
  } else if (spirv_debugInfo) {
    options.spirv.debugInfo = denox::spirv::SpirvDebugInfoLevel::Enable;
  }

  for (auto &[name, interface] : tensorDescriptors) {
    interface.name = name;
    options.interfaceDescriptors.push_back(interface);
  }

  for (auto &[name, value] : assumptions) {
    options.assumptions.valueAssumptions.emplace_back(name, value);
  }


  if (!input.has_value()) {
    throw ParseError(fmt::format("requires --input argument"));
  }

  IOEndpoint out = output.value_or([&]() -> IOEndpoint {
    switch (input->kind()) {
    case IOEndpointKind::Path:
      return input->path().with_filename("a");
    case IOEndpointKind::Pipe:
      return Pipe{};
    }
    denox::diag::unreachable();
  }());


  return InferAction{
      .model = std::move(model),
      .input = *input,
      .output = out,
      .deviceName = deviceName,
      .apiVersion = apiVersion,
      .database = std::move(database),
      .options = options,
  };
}

Action parse_version(std::span<const Token>) { return Action::version(); }

Action parse_help(std::span<const Token>) {
  return HelpAction(HelpScope::Global);
}
Action parse_dumpcsv(std::span<const Token> tokens) {
  if (tokens.size() < 1) {
    throw ParseError(fmt::format(
        "dumpcsv expects a database as the first positional argument"));
  }

  const auto &dbToken = tokens.front();
  if (dbToken.kind() != TokenKind::Literal || !dbToken.literal().is_path()) {
    throw ParseError(fmt::format(
        "expected database path as first positional argument, got {}",
        dbToken));
  }

  DbArtefact database(dbToken.literal().as_path());

  denox::memory::optional<IOEndpoint> output;

  uint32_t i = 1;
  while (i < tokens.size()) {
    uint32_t jump = 0;

    auto tail = denox::memory::span{tokens.begin() + i, tokens.end()};

    // --- positional arguments are not allowed here ---
    if (tokens[i].kind() == TokenKind::Literal ||
        tokens[i].kind() == TokenKind::Pipe) {
      throw ParseError(
          fmt::format("unexpected argument {}", describe_token(tokens[i])));
    }

    if ((jump = parse_output(tail, &output))) {
      i += jump;
      continue;
    }

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
  }

  IOEndpoint csv = output.value_or(Pipe{});

  return DumpCsvAction{
      .database = std::move(database),
      .csv = csv,
  };
}
