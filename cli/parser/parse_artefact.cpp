#include "parser/parse_artefact.hpp"
#include "io/File.hpp"
#include "io/IOEndpoint.hpp"
#include "onnx.pb.h"
#include <db.h>
#include <dnx.h>
#include <fmt/base.h>
#include <iterator>
#include <stdexcept>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <onnx.pb.h>

static bool is_onnx_model(std::span<const std::byte> data) {
  onnx::ModelProto model;

  google::protobuf::io::ArrayInputStream stream(data.data(),
                                                static_cast<int>(data.size()));

  google::protobuf::io::CodedInputStream coded(&stream);

  // Allow large models
  coded.SetTotalBytesLimit(std::numeric_limits<int>::max());

  if (!model.ParseFromCodedStream(&coded)) {
    return false;
  }

  // Optional sanity checks
  if (!model.has_graph()) {
    return false;
  }

  if (model.ir_version() == 0) {
    return false;
  }

  return true;
}

static bool is_dnx_model(std::span<const std::byte> data) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
  return denox::dnx::VerifyModelBuffer(verifier);
}

static bool is_db(std::span<const std::byte> data) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
  return denox::db::VerifyDbBuffer(verifier);
}

static std::optional<Artefact> resolve_artefact(std::span<const std::byte> data,
                                                IOEndpoint endpoint) {
  const bool isDNX = is_dnx_model(data);
  const bool isONNX = is_onnx_model(data);
  const bool isDB = is_db(data);

  fmt::println("isDNX: {}", isDNX);
  fmt::println("isONNX: {}", isONNX);
  fmt::println("isDB: {}", isDB);

  const int count = (isDNX ? 1 : 0) + (isONNX ? 1 : 0) + (isDB ? 1 : 0);

  if (count == 0) {
    return std::nullopt;
  }

  if (count > 1) {
    throw std::runtime_error("input buffer matches multiple artefact formats; "
                             "input is ambiguous or corrupted");
  }

  if (isDNX) {
    return Artefact(
        DnxArtefact{.endpoint = endpoint,
                    .data = std::vector<std::byte>(data.begin(), data.end())});
  }

  if (isONNX) {
    return Artefact(
        OnnxArtefact{.endpoint = endpoint,
                     .data = std::vector<std::byte>(data.begin(), data.end())});
  }

  if (isDB) {
    return Artefact(DbArtefact{endpoint});
  }

  std::abort(); // logically unreachable
}

std::optional<Artefact> parse_artefact(const Token &token) {
  switch (token.kind()) {
  case TokenKind::Command:
  case TokenKind::Option:
    return std::nullopt;
  case TokenKind::Pipe: {
    const auto &pipe = token.pipe();
    std::vector<std::byte> data = pipe.read_all();
    if (data.empty()) {
      return std::nullopt;
    }
    auto artefact = resolve_artefact(data, pipe);
    if (artefact.has_value() && artefact->kind() == ArtefactKind::Database) {
      throw std::runtime_error("database files cannot be piped");
    }
    return artefact;
  }
  case TokenKind::Literal: {
    const auto &literal = token.literal();
    if (!literal.is_path()) {
      // don't interpret this as a artefact!
      return std::nullopt;
    }
    const Path &path = literal.as_path();
    if (!path.exists()) {
      if (path.extension() == ".db") {
        return Artefact(DbArtefact(IOEndpoint(path)));
      }
      return std::nullopt;
    }
    auto file = File::open(path, File::OpenMode::Read);
    size_t size = file.size();
    std::vector<std::byte> data(size);
    file.read_exact(data);
    return resolve_artefact(data, path);
  }
  }
  throw std::runtime_error("unreachable");
}
