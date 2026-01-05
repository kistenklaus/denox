#include "denox/cli/parser/parse_commands.hpp"
#include "denox/cli/alloc/monotone_alloc.hpp"
#include "denox/cli/io/Pipe.hpp"
#include "denox/cli/parser/errors.hpp"
#include "denox/cli/parser/parse_artefact.hpp"
#include "denox/cli/parser/parse_options.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
// #include "denox/cli/parser/parse_options.hpp"
#include <absl/strings/str_format.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

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

    if ((jump = parse_feature_fusion(tail,
                                     &options.features.enableConvReluFusion))) {
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

    // --- nothing matched ---
    throw ParseError(
        fmt::format("invalid option {}", describe_token(tokens[i])));
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

  IOEndpoint out = output.value_or([&]() -> IOEndpoint {
    switch (input.endpoint.kind()) {
    case IOEndpointKind::Path:
      return input.endpoint.path().with_extension("dnx");
    case IOEndpointKind::Pipe:
      return denox::io::Path::cwd() / "a.dnx";
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
  std::optional<OnnxArtefact> model;
  std::optional<DbArtefact> database;

  denox::diag::not_implemented();
}

Action parse_bench(std::span<const Token> tokens) {
  denox::diag::not_implemented();
}

Action parse_infer(std::span<const Token> tokens) {
  denox::diag::not_implemented();
}

Action parse_version(std::span<const Token> tokens) {
  return Action::version();
}

Action parse_help(std::span<const Token> tokens) {
  return HelpAction(HelpScope::Global);
}
