#include "denox/cli/parser/parse_options.hpp"
#include "denox/cli/alloc/monotone_alloc.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/errors.hpp"
#include "denox/cli/parser/lex/tokens/option.hpp"
#include "denox/cli/parser/parse_artefact.hpp"
#include "denox/common/ValueName.hpp"
#include "denox/common/ValueSpec.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/container/vector.hpp"
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

uint32_t
parse_use_descriptor_sets(std::span<const Token> tokens,
                          denox::compiler::DescriptorPolicies *policies) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::UseDescriptorSets) {
    return 0;
  }
  if (tokens.size() < 6) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  const auto &intok = tokens[1];
  const auto &outok = tokens[2];
  const auto &patok = tokens[3];
  const auto &retok = tokens[4];
  const auto &wrtok = tokens[5];

  if (intok.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (outok.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (patok.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (retok.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (wrtok.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  const auto &inlit = intok.literal();
  const auto &oulit = outok.literal();
  const auto &palit = patok.literal();
  const auto &relit = retok.literal();
  const auto &wrlit = wrtok.literal();

  if (!inlit.is_unsigned_int()) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (!oulit.is_unsigned_int()) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (!palit.is_unsigned_int()) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (!relit.is_unsigned_int()) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  if (!wrlit.is_unsigned_int()) {
    throw ParseError(
        fmt::format("--use-descriptor-sets expects 5 arguments <input-set> "
                    "<output-set> <param-set> <read-set> <write-set>"));
  }
  policies->inputPolicy.set = static_cast<uint32_t>(inlit.as_unsigned_int());
  policies->outputPolicy.set = static_cast<uint32_t>(oulit.as_unsigned_int());
  policies->paramPolicy.set = static_cast<uint32_t>(palit.as_unsigned_int());
  policies->readPolicy.set = static_cast<uint32_t>(relit.as_unsigned_int());
  policies->writePolicy.set = static_cast<uint32_t>(wrlit.as_unsigned_int());
  return 6;
}

uint32_t parse_assume(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string, int64_t> &assumptions) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Assume) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 1) {
      throw ParseError(fmt::format("--assume expects \"X=123\""));
    }
    const auto &valueNameToken = tokens[i];
    if (valueNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--assume expects \"X=123\""));
    }
    const auto &valueToken = tokens[i + 1];
    if (valueToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--assume expects \"X=123\""));
    }
    if (!valueToken.literal().is_signed_int()) {
      throw ParseError(fmt::format("--assume expects \"X=123\""));
    }
    denox::memory::string name;
    name = valueNameToken.literal().view();
    assumptions[name] = valueToken.literal().as_signed_int();
    i += 2;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);
}

uint32_t parse_specialize(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string, int64_t> &assumptions) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Specialize) {
    return 0;
  }
  size_t i = 1;
  int64_t r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  while (r > 0) {
    if (tokens[i].kind() == TokenKind::Option) {
      return static_cast<uint32_t>(i);
    }
    if (r <= 1) {
      throw ParseError(fmt::format("--specialize expects \"X=123\""));
    }
    const auto &valueNameToken = tokens[i];
    if (valueNameToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--specialize expects \"X=123\""));
    }
    const auto &valueToken = tokens[i + 1];
    if (valueToken.kind() != TokenKind::Literal) {
      throw ParseError(fmt::format("--specialize expects \"X=123\""));
    }
    if (!valueToken.literal().is_signed_int()) {
      throw ParseError(fmt::format("--specialize expects \"X=123\""));
    }
    denox::memory::string name;
    name = valueNameToken.literal().view();
    assumptions[name] = valueToken.literal().as_signed_int();
    i += 2;
    r = static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(i);
  }
  return static_cast<uint32_t>(i);
}

uint32_t parse_samples(std::span<const Token> tokens, uint32_t *samples) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::Samples) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw ParseError(fmt::format("--samples expects one integral"));
  }
  const auto &samplesToken = tokens[1];
  if (samplesToken.kind() != TokenKind::Literal) {
    throw ParseError(fmt::format("--samples expects one integral"));
  }
  if (!samplesToken.literal().is_unsigned_int()) {
    throw ParseError(fmt::format("--samples expects one integral"));
  }
  if (samples) {
    *samples = static_cast<uint32_t>(samplesToken.literal().as_unsigned_int());
  }
  return 2;
}

uint32_t parse_relative_error(std::span<const Token> tokens,
                              float *relative_error) {
  if (tokens.empty()) {
    return 0;
  }
  const Token &head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  if (head.option() != OptionToken::RelativeError) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw ParseError(
        fmt::format("--relative-error expects floating-point number"));
  }
  const auto &samplesToken = tokens[1];
  if (samplesToken.kind() != TokenKind::Literal) {
    throw ParseError(
        fmt::format("--relative-error expects floating-point number"));
  }
  if (!samplesToken.literal().is_float()) {
    throw ParseError(
        fmt::format("--relative-error expects floating-point number"));
  }
  if (relative_error) {
    *relative_error = samplesToken.literal().as_float();
  }
  return 2;
}
uint32_t parse_input(std::span<const Token> tokens,
                     denox::memory::optional<IOEndpoint> *in) {
  if (tokens.empty()) {
    return 0;
  }
  auto head = tokens.front();
  if (head.kind() != TokenKind::Option) {
    return 0;
  }
  const auto &option = head.option();
  if (option != OptionToken::Input) {
    return 0;
  }
  if (tokens.size() < 2) {
    throw ParseError("option '--input' requires an argument");
  }
  const auto arg = tokens[1];
  if (arg.kind() == TokenKind::Literal) {
    const auto literal = arg.literal();
    if (!literal.is_path()) {
      throw ParseError("invalid argument to '--input': expected a path or '-'");
    }
    const auto path = literal.as_path();
    if (path.is_dir()) {
      throw ParseError(fmt::format(
          "invalid argument to '--input': '{}' is a directory", path));
    }
    if (in != nullptr) {
      *in = IOEndpoint{path};
    }
    return 2;
  } else if (arg.kind() == TokenKind::Pipe) {
    if (in != nullptr) {
      *in = IOEndpoint{Pipe{}};
    }
    return 2;
  } else if (arg.kind() == TokenKind::Option) {
    throw ParseError("option '--input' requires an argument");
  } else {
    throw ParseError("invalid argument to '--input': expected a path or '-'");
  }
}

uint32_t parse_help(std::span<const Token> tokens, bool *help) {
  if (tokens.empty()) {
    return 0;
  }
  const auto &head = tokens.front();
  if (head.kind() != TokenKind::Command) {
    return 0;
  }
  if (head.command() != CommandToken::Help) {
    return 0;
  }
  if (help) {
    *help = true;
  }
  return 1;
}
