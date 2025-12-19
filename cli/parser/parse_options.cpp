#include "parser/parse_options.hpp"
#include "alloc/monotone_alloc.hpp"
#include "denox/compiler.hpp"
#include "io/File.hpp"
#include "io/IOEndpoint.hpp"
#include "parser/errors.hpp"
#include "parser/parse_artefact.hpp"
#include <fmt/base.h>
#include <google/protobuf/extension_set.h>
#include <stdexcept>

static uint32_t parse_bool_flag(std::span<const Token> tokens,
                                OptionToken expected, bool *out,
                                std::string_view name) {
  if (tokens.empty()) {
    return 0;
  }

  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }

  if (head.option() != expected) {
    return 0;
  }

  // Flag present without value → true
  if (tokens.size() < 2 || tokens[1].kind() != TokenKind::Literal) {
    if (out) {
      *out = true;
    }
    return 1;
  }

  const LiteralToken &lit = tokens[1].literal();
  if (!lit.is_bool()) {
    throw ParseError(
        fmt::format("invalid value for --{} (expected true or false)", name));
  }

  if (out) {
    *out = lit.as_bool();
  }
  return 2;
}

uint32_t parse_spirv_optimize(std::span<const Token> t, bool *v) {
  return parse_bool_flag(t, OptionToken::SpirvOptimize, v, "spirv-optimize");
}

uint32_t parse_spirv_non_semantic_debug_info(std::span<const Token> t,
                                             bool *v) {
  return parse_bool_flag(t, OptionToken::SpirvNonSemanticDebugInfo, v,
                         "spirv-non-semantic-debug-info");
}

uint32_t parse_spirv_debug_info(std::span<const Token> t, bool *v) {
  return parse_bool_flag(t, OptionToken::SpirvDebugInfo, v, "spirv-debug-info");
}

uint32_t parse_verbose(std::span<const Token> t, bool *v) {
  return parse_bool_flag(t, OptionToken::Verbose, v, "verbose");
}

uint32_t parse_quiet(std::span<const Token> t, bool *v) {
  return parse_bool_flag(t, OptionToken::Quiet, v, "quiet");
}

static uint32_t parse_feature_flag(std::span<const Token> tokens,
                                   OptionToken expected,
                                   denox::FeatureState *out,
                                   std::string_view name) {
  if (tokens.empty())
    return 0;

  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option)
    return 0;

  if (head.option() != expected)
    return 0;

  // Flag present without value → true
  if (tokens.size() < 2 || tokens[1].kind() != TokenKind::Literal) {
    if (out) {
      *out = denox::FeatureState::Require;
    }
    return 1;
  }

  const LiteralToken &lit = tokens[1].literal();
  if (!lit.is_bool()) {
    throw ParseError(
        fmt::format("invalid value for --{} (expected true or false)", name));
  }

  if (out) {
    bool b = lit.as_bool();
    if (b) {
      *out = denox::FeatureState::Require;
    } else {
      *out = denox::FeatureState::Disable;
    }
  }
  return 2;
}

uint32_t parse_feature_coopmat(std::span<const Token> t,
                               denox::FeatureState *v) {
  return parse_feature_flag(t, OptionToken::FeatureCoopmat, v,
                            "feature-coopmat");
}

uint32_t parse_feature_fusion(std::span<const Token> t,
                              denox::FeatureState *v) {
  return parse_feature_flag(t, OptionToken::FeatureFusion, v, "feature-fusion");
}

uint32_t parse_feature_memory_concat(std::span<const Token> t,
                                     denox::FeatureState *v) {
  return parse_feature_flag(t, OptionToken::FeatureMemoryConcat, v,
                            "feature-memory-concat");
}

uint32_t parse_output(std::span<const Token> tokens,
                      std::optional<IOEndpoint> *out) {
  if (tokens.empty()) {
    return 0;
  }
  auto head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::Output) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw ParseError("option '-o' requires an argument");
  }
  const auto arg = tokens[1];
  if (arg.kind() == TokenKind::Literal) {
    const auto literal = arg.literal();
    if (!literal.is_path()) {
      throw ParseError("invalid argument to '-o': expected a path or '-'");
    }
    const auto path = literal.as_path();
    if (path.is_dir()) {
      throw ParseError(
          fmt::format("invalid argument to '-o': '{}' is a directory", path));
    }
    if (out != nullptr) {
      *out = IOEndpoint{path};
    }
    return 2;
  } else if (arg.kind() == TokenKind::Pipe) {
    if (out != nullptr) {
      *out = IOEndpoint{Pipe{}};
    }
    return 2;
  } else if (arg.kind() == TokenKind::Option) {
    throw ParseError("option '-o' requires an argument");
  } else {
    throw ParseError("invalid argument to '-o': expected a path or '-'");
  }
}

uint32_t parse_device(std::span<const Token> tokens, const char **device) {
  if (tokens.empty()) {
    return 0;
  }
  const auto &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::Device) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw std::runtime_error("option '--device' requires an argument");
  }
  const auto pat = tokens[1];
  if (pat.kind() == TokenKind::Option || pat.kind() == TokenKind::Command) {
    throw ParseError("option '--device' requires an argument");
  }
  if (pat.kind() != TokenKind::Literal) {
    throw std::runtime_error(
        "invalid argument to '--device': expected a device name");
  }
  if (device != nullptr) {
    const auto lit = pat.literal();
    char *cstr = static_cast<char *>(mono_alloc(lit.view().size() + 1));
    std::memset(cstr, 0, lit.view().size() + 1);
    std::memcpy(cstr, lit.view().data(), lit.view().size());

    *device = cstr;
  }
  return 2;
}

uint32_t parse_target_env(std::span<const Token> tokens,
                          denox::VulkanApiVersion *target_env) {
  if (tokens.empty()) {
    return 0;
  }
  const auto &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::TargetEnv) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw std::runtime_error("option '--target-env' requires an argument");
  }
  const auto &pat = tokens[1];
  if (pat.kind() == TokenKind::Option || pat.kind() == TokenKind::Command) {
    throw ParseError("option '--target-env' requires an argument");
  }
  if (pat.kind() != TokenKind::Literal) {
    throw std::runtime_error("invalid argument to '--target-env': expected "
                             "vulkan1.0|vulkan1.1|...");
  }
  const auto &lit = pat.literal();
  if (!lit.is_target_env()) {
    throw std::runtime_error("invalid argument to '--target-env': expected a "
                             "vulkan1.0|vulkan1.1|...");
  }
  if (target_env) {
    *target_env = lit.as_target_env();
  }
  return 2;
}

static denox::BufferDescription &get_or_create_desc(
    std::unordered_map<std::string, denox::BufferDescription> &map,
    std::string_view name) {
  auto it = map.find(std::string(name));
  if (it == map.end()) {
    it = map.emplace(std::string(name), denox::BufferDescription{}).first;
  }
  return it->second;
}

static uint32_t
parse_layout(std::span<const Token> tokens, OptionToken expected,
             std::unordered_map<std::string, denox::BufferDescription> &descs,
             std::string_view opt_name) {
  if (tokens.empty())
    return 0;

  if (tokens.front().kind() != TokenKind::Option ||
      tokens.front().option() != expected)
    return 0;

  size_t i = 1;

  auto parse_value = [&](std::string_view key, const LiteralToken &lit) {
    if (!lit.is_layout()) {
      throw ParseError(fmt::format(
          "invalid value for '--{}': expected layout (HWC, CHW, RGB, ...)",
          opt_name));
    }
    auto &desc = get_or_create_desc(descs, key);
    desc.layout = lit.as_layout();
  };

  if (i >= tokens.size())
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));

  // Case 1: single literal → default "*"
  if (tokens.size() - i == 1) {
    if (tokens[i].kind() != TokenKind::Literal)
      throw ParseError(fmt::format("invalid argument to '--{}'", opt_name));

    parse_value("*", tokens[i].literal());
    return 2;
  }

  // Case 2: named pairs
  while (i + 1 < tokens.size()) {
    if (tokens[i].kind() != TokenKind::Literal ||
        tokens[i + 1].kind() != TokenKind::Literal)
      break;

    std::string_view name = tokens[i].literal().view();
    if (tokens[i].literal().is_layout()) {
      throw ParseError(fmt::format("invalid argument to '--{}'", opt_name));
    }
    const auto &value_lit = tokens[i + 1].literal();

    parse_value(name, value_lit);
    i += 2;
  }

  if (i == 1) {
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));
  }

  return static_cast<uint32_t>(i);
}

uint32_t parse_input_layout(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs) {
  return parse_layout(tokens, OptionToken::InputLayout, inputs, "input-layout");
}

uint32_t parse_output_layout(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &outputs) {
  return parse_layout(tokens, OptionToken::OutputLayout, outputs,
                      "output-layout");
}

static uint32_t
parse_storage(std::span<const Token> tokens, OptionToken expected,
              std::unordered_map<std::string, denox::BufferDescription> &descs,
              std::string_view opt_name) {
  if (tokens.empty())
    return 0;

  if (tokens.front().kind() != TokenKind::Option ||
      tokens.front().option() != expected)
    return 0;

  size_t i = 1;

  auto set_storage = [&](std::string_view key, const LiteralToken &lit) {
    if (!lit.is_storage()) {
      throw ParseError(fmt::format(
          "invalid value for '--{}': expected ssbo|image|sampler", opt_name));
    }

    auto &d = descs[std::string(key)];
    d.storage = lit.as_storage();
  };

  if (i >= tokens.size()) {
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));
  }

  // Case 1: single literal → applies to "*"
  if (tokens.size() - i == 1) {
    if (tokens[i].kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("invalid argument to '--{}'", opt_name));
    }

    set_storage("*", tokens[i].literal());
    return 2;
  }

  // Case 2: named pairs
  while (i + 1 < tokens.size()) {
    if (tokens[i].kind() != TokenKind::Literal ||
        tokens[i + 1].kind() != TokenKind::Literal)
      break;

    std::string_view name = tokens[i].literal().view();
    const auto &value = tokens[i + 1].literal();

    set_storage(name, value);
    i += 2;
  }

  if (i == 1) {
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));
  }

  return static_cast<uint32_t>(i);
}

uint32_t parse_input_storage(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs) {
  return parse_storage(tokens, OptionToken::InputStorage, inputs,
                       "input-storage");
}

uint32_t parse_output_storage(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &outputs) {
  return parse_storage(tokens, OptionToken::OutputStorage, outputs,
                       "input-storage");
}

static uint32_t
parse_type(std::span<const Token> tokens, OptionToken expected,
           std::unordered_map<std::string, denox::BufferDescription> &inputs,
           std::string_view opt_name) {
  if (tokens.empty())
    return 0;

  if (tokens.front().kind() != TokenKind::Option ||
      tokens.front().option() != expected)
    return 0;

  size_t i = 1;

  auto set_type = [&](std::string_view key, const LiteralToken &lit) {
    if (!lit.is_dtype()) {
      throw ParseError(fmt::format(
          "invalid value for '--{}': expected f32|f16|u8|i8", opt_name));
    }

    auto &desc = inputs[std::string(key)];
    desc.dtype = lit.as_dtype();
  };

  if (i >= tokens.size()) {
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));
  }

  // Case 1: single literal → default "*"
  if (tokens.size() - i == 1) {
    if (tokens[i].kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("invalid argument to '--{}'", opt_name));
    }

    set_type("*", tokens[i].literal());
    return 2;
  }

  // Case 2: named pairs
  while (i + 1 < tokens.size()) {
    if (tokens[i].kind() != TokenKind::Literal ||
        tokens[i + 1].kind() != TokenKind::Literal)
      break;

    std::string_view name = tokens[i].literal().view();
    const auto &value = tokens[i + 1].literal();

    set_type(name, value);
    i += 2;
  }

  if (i == 1) {
    throw ParseError(
        fmt::format("option '--{}' requires an argument", opt_name));
  }

  return static_cast<uint32_t>(i);
}

uint32_t parse_input_type(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs) {
  return parse_type(tokens, OptionToken::InputType, inputs, "input-type");
}

uint32_t parse_output_type(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs) {
  return parse_type(tokens, OptionToken::OutputType, inputs, "output-type");
}

static denox::Extent parse_extent(const LiteralToken &lit) {
  denox::Extent e{};
  e.name = nullptr;
  e.value = 0;

  // value-only (e.g. 1080)
  if (lit.is_unsigned_int()) {
    e.value = lit.as_unsigned_int();
    return e;
  }

  // name or name=value
  const auto sv = lit.view();
  const auto eq = sv.find('=');

  auto alloc_name = [&](std::string_view n) {
    char *p = static_cast<char *>(mono_alloc(n.size() + 1));
    std::memcpy(p, n.data(), n.size());
    p[n.size()] = '\0';
    return p;
  };

  if (eq == std::string_view::npos) {
    e.name = alloc_name(sv);
    return e;
  }

  // name=value
  std::string_view name = sv.substr(0, eq);
  std::string_view val = sv.substr(eq + 1);

  e.name = alloc_name(name);

  uint64_t v{};
  auto res = std::from_chars(val.data(), val.data() + val.size(), v);
  if (res.ec != std::errc{} || res.ptr != val.data() + val.size()) {
    throw ParseError("invalid shape value");
  }

  e.value = v;
  return e;
}

static uint32_t parse_shape(
    std::span<const Token> tokens, OptionToken expected,
    std::unordered_map<std::string, denox::BufferDescription> &descriptors,
    std::string_view opt_name) {

  if (tokens.empty())
    return 0;

  if (tokens.front().kind() != TokenKind::Option ||
      tokens.front().option() != expected)
    return 0;

  size_t i = 1;
  size_t literal_count = 0;

  while (i + literal_count < tokens.size() &&
         tokens[i + literal_count].kind() == TokenKind::Literal) {
    ++literal_count;
  }

  if (literal_count == 0) {
    throw ParseError(fmt::format("option '--{}' requires arguments", opt_name));
  }

  // ---- Case 1: default shape (exactly 3 literals) ----
  if (literal_count == 3) {
    denox::Shape shape{};
    shape.height = parse_extent(tokens[i + 0].literal());
    shape.width = parse_extent(tokens[i + 1].literal());
    shape.channels = parse_extent(tokens[i + 2].literal());

    descriptors["*"].shape = shape;
    return 1 + 3;
  }

  // ---- Case 2: named shapes (N * 4 literals) ----
  if (literal_count % 4 != 0) {
    throw ParseError(fmt::format("invalid syntax for '--{}'", opt_name));
  }

  uint32_t consumed = 1;

  while (i + 3 < tokens.size() && tokens[i].kind() == TokenKind::Literal) {

    const auto &name_lit = tokens[i].literal();
    std::string name{name_lit.view()};

    denox::Shape shape{};
    shape.height = parse_extent(tokens[i + 1].literal());
    shape.width = parse_extent(tokens[i + 2].literal());
    shape.channels = parse_extent(tokens[i + 3].literal());

    descriptors[name].shape = shape;

    i += 4;
    consumed += 4;
  }

  return consumed;
}

uint32_t
parse_input_shape(std::span<const Token> tokens,
                  std::unordered_map<std::string, denox::BufferDescription>
                      &input_descriptors) {
  return parse_shape(tokens, OptionToken::InputShape, input_descriptors,
                     "input-shape");
}

uint32_t
parse_output_shape(std::span<const Token> tokens,
                   std::unordered_map<std::string, denox::BufferDescription>
                       &output_descriptors) {
  return parse_shape(tokens, OptionToken::OutputShape, output_descriptors,
                     "output-shape");
}

uint32_t parse_database(std::span<const Token> tokens,
                        std::optional<DbArtefact> *db) {
  if (tokens.empty()) {
    return 0;
  }
  const auto &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::Database) {
    return false;
  }
  if (tokens.size() < 2) {
    throw ParseError("option '--database' requires an argument");
  }
  const auto &arg = tokens[1];
  if (arg.kind() == TokenKind::Command) {
    throw ParseError("option '--database' requires an argument");
  }

  if (arg.kind() != TokenKind::Literal) {
    throw ParseError(fmt::format(
        "invalid argument to '--database': expected database path"));
  }
  const auto &lit = arg.literal();
  if (!lit.is_path()) {
    throw ParseError(fmt::format(
        "invalid argument to '--database': expected database path"));
  }
  const auto &path = lit.as_path();
  if (!path.exists()) {
    throw ParseError(fmt::format("file not found: '{}'", path));
  }
  if (path.is_dir()) {
    throw ParseError(fmt::format("'{}' is a directory", path));
  }

  File file = File::open(path, File::OpenMode::Read);
  std::vector<std::byte> data(file.size());
  file.read_exact(data);
  bool isDB = is_db(data);
  if (!isDB) {
    throw ParseError(fmt::format("'{}' is not a valid database", path));
  }

  if (db) {
    *db = DbArtefact{IOEndpoint(path)};
  }
  return 2;
}

uint32_t parse_assume(std::span<const Token> tokens,
                      std::vector<denox::OptimizationAssumption> &assumptions) {
  if (tokens.empty()) {
    return 0;
  }
  const auto &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::Assume) {
    return 0;
  }
  uint32_t step = 1;
  for (uint32_t i = 1; i < tokens.size(); i += 2) {
    uint32_t remaining = static_cast<uint32_t>(tokens.size()) - i;
    if (remaining < 2) {
      break;
    }
    const auto &var = tokens[i];
    const auto &val = tokens[i + 1];
    if (var.kind() != TokenKind::Literal || val.kind() != TokenKind::Literal) {
      break;
    }
    const auto name = var.literal();
    if (name.is_unsigned_int() || name.is_signed_int()) {
      throw ParseError("invalid argument to '--assume': expected variable "
                       "assignments (e.g. H=10 W=20 ...)");
    }
    const auto value = val.literal();
    if (!value.is_unsigned_int()) {
      throw ParseError("invalid argument to '--assume': expected variable "
                       "assignments (e.g. H=10 W=20 ...)");
    }
    char *cstr = static_cast<char *>(mono_alloc(name.view().size() + 1));
    std::memset(cstr, 0, name.view().size() + 1);
    std::memcpy(cstr, name.view().data(), name.view().size());

    assumptions.push_back(denox::OptimizationAssumption{
        .valueName = cstr,
        .value = value.as_unsigned_int(),
    });
    step += 2;
  }
  if (step < tokens.size() && tokens[step].kind() == TokenKind::Literal) {
    throw std::runtime_error(
        "option '--assume' requires a even amount of arguments");
  }
  return step;
}
