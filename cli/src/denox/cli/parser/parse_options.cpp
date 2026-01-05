#include "denox/cli/parser/parse_options.hpp"
#include "denox/cli/alloc/monotone_alloc.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/errors.hpp"
#include "denox/cli/parser/parse_artefact.hpp"
#include "denox/common/ValueName.hpp"
#include "denox/common/ValueSpec.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/container/vector.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
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
                                   OptionToken expected, bool *out,
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
    bool b = lit.as_bool();
    if (b) {
      *out = true;
    } else {
      *out = false;
    }
  }
  return 2;
}

uint32_t parse_feature_coopmat(std::span<const Token> t, bool *v) {
  return parse_feature_flag(t, OptionToken::FeatureCoopmat, v,
                            "feature-coopmat");
}

uint32_t parse_feature_fusion(std::span<const Token> t, bool *v) {
  return parse_feature_flag(t, OptionToken::FeatureFusion, v, "feature-fusion");
}

uint32_t parse_feature_memory_concat(std::span<const Token> t, bool *v) {
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

uint32_t
parse_device(std::span<const Token> tokens,
             denox::memory::optional<denox::memory::string> *deviceName) {
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
  if (deviceName != nullptr) {
    *deviceName = pat.literal().view();
  }
  return 2;
}

uint32_t parse_target_env(std::span<const Token> tokens,
                          denox::ApiVersion *target_env) {
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

  auto file = denox::io::File::open(path, denox::io::File::OpenMode::Read);
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

uint32_t
parse_shape(std::span<const Token> tokens,
            denox::memory::hash_map<denox::memory::string,
                                    denox::compiler::InterfaceTensorDescriptor>
                &tensorDescriptors) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Shape) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 3) {
      throw ParseError(fmt::format("--shape expects \"tensor-name=X:Y:Z\""));
    }
    const auto &tensorNameToken = tokens[i];
    if (tensorNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--shape expects \"tensor-name=X:Y:Z\""));
    }
    const auto &heightToken = tokens[i + 1];
    if (heightToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--shape expects \"tensor-name=X:Y:Z\""));
    }
    const auto &widthToken = tokens[i + 2];
    if (widthToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--shape expects \"tensor-name=X:Y:Z\""));
    }
    const auto &channelToken = tokens[i + 3];
    if (channelToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--shape expects \"tensor-name=X:Y:Z\""));
    }

    denox::memory::string name;
    name = tensorNameToken.literal().view();
    auto &descriptor = tensorDescriptors[name];
    descriptor.heightValueName = heightToken.literal().view();
    descriptor.widthValueName = widthToken.literal().view();
    descriptor.channelValueName = channelToken.literal().view();

    i += 4;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);
}

uint32_t parse_storage(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string,
                            denox::compiler::InterfaceTensorDescriptor>
        &tensorDescriptors) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Storage) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 1) {
      throw ParseError(fmt::format("--storage expects \"ssbo,ssi,image,...\""));
    }
    const auto &tensorNameToken = tokens[i];
    if (tensorNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--storage expects \"ssbo,ssi,image,...\""));
    }
    const auto &storageToken = tokens[i + 1];
    if (storageToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--storage expects \"ssbo,ssi,image,...\""));
    }
    if (!storageToken.literal().is_storage()) {
      throw ParseError(fmt::format("--storage expects \"ssbo,ssi,image,...\""));
    }

    denox::memory::string name;
    name = tensorNameToken.literal().view();
    auto &descriptor = tensorDescriptors[name];
    descriptor.storage = storageToken.literal().as_storage();

    i += 2;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);
}
uint32_t
parse_format(std::span<const Token> tokens,
             denox::memory::hash_map<denox::memory::string,
                                     denox::compiler::InterfaceTensorDescriptor>
                 &tensorDescriptors) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Format) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 1) {
      throw ParseError(fmt::format("--format expects \"hwc,chw,rgb,...\""));
    }
    const auto &tensorNameToken = tokens[i];
    if (tensorNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--format expects \"hwc,chw,rgb,...\""));
    }
    const auto &storageToken = tokens[i + 1];
    if (storageToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--format expects \"hwc,chw,rgb,...\""));
    }
    if (!storageToken.literal().is_format()) {
      throw ParseError(fmt::format("--format expects \"hwc,chw,rgb,...\""));
    }

    denox::memory::string name;
    name = tensorNameToken.literal().view();
    auto &descriptor = tensorDescriptors[name];
    descriptor.format = storageToken.literal().as_format();

    i += 2;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);
}

uint32_t
parse_type(std::span<const Token> tokens,
           denox::memory::hash_map<denox::memory::string,
                                   denox::compiler::InterfaceTensorDescriptor>
               &tensorDescriptors) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Type) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 1) {
      throw ParseError(fmt::format("--type expects \"f16,f32,...\""));
    }
    const auto &tensorNameToken = tokens[i];
    if (tensorNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--type expects \"f16,f32,...\""));
    }
    const auto &storageToken = tokens[i + 1];
    if (storageToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--type expects \"f16,f32,...\""));
    }
    if (!storageToken.literal().is_dtype()) {
      throw ParseError(fmt::format("--type expects \"f16,f32,...\""));
    }

    denox::memory::string name;
    name = tensorNameToken.literal().view();
    auto &descriptor = tensorDescriptors[name];
    descriptor.dtype = storageToken.literal().as_dtype();

    i += 2;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);

}
