#include "denox/cli/parser/parse_artefact.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/io/fs/File.hpp"
#include "onnx.pb.h"
#include <db.h>
#include <dnx.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <onnx.pb.h>

bool is_onnx_model(std::span<const std::byte> data) {
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

bool is_dnx_model(std::span<const std::byte> data) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
  return denox::dnx::VerifyModelBuffer(verifier);
}

bool is_db(std::span<const std::byte> data) {
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size());
  return denox::db::VerifyDbBuffer(verifier);
}

static ArtefactParseResult resolve_artefact(std::span<const std::byte> data,
                                            IOEndpoint endpoint) {
  const bool isDNX = is_dnx_model(data);
  const bool isONNX = is_onnx_model(data);
  const bool isDB = is_db(data);

  const int count = (isDNX ? 1 : 0) + (isONNX ? 1 : 0) + (isDB ? 1 : 0);

  if (count == 0) {
    return ArtefactParseResult::failure(ArtefactParseError::UnrecognizedFormat);
  }

  if (count > 1) {
    return ArtefactParseResult::failure(ArtefactParseError::AmbiguousFormat);
  }

  if (isDNX) {
    return ArtefactParseResult::success(Artefact(
        DnxArtefact{.endpoint = endpoint,
                    .data = std::vector<std::byte>(data.begin(), data.end())}));
  }

  if (isONNX) {
    return ArtefactParseResult::success(Artefact(OnnxArtefact{
        .endpoint = endpoint,
        .data = std::vector<std::byte>(data.begin(), data.end())}));
  }

  // Database never owns bytes
  return ArtefactParseResult::success(Artefact(DbArtefact{endpoint}));
}

ArtefactParseResult parse_artefact(const Token &token) {
  switch (token.kind()) {

  case TokenKind::Command:
  case TokenKind::Option:
    return ArtefactParseResult::failure(ArtefactParseError::NotAnArtefactToken);

  case TokenKind::Pipe: {
    std::vector<std::byte> data = Pipe{}.read_all();

    if (data.empty()) {
      return ArtefactParseResult::failure(
          ArtefactParseError::UnrecognizedFormat);
    }

    auto res = resolve_artefact(data, IOEndpoint(Pipe{}));

    if (res.artefact && res.artefact->kind() == ArtefactKind::Database) {
      return ArtefactParseResult::failure(ArtefactParseError::DatabasePiped);
    }

    return res;
  }

  case TokenKind::Literal: {
    const LiteralToken &lit = token.literal();

    if (!lit.is_path()) {
      return ArtefactParseResult::failure(
          ArtefactParseError::NotAnArtefactToken);
    }

    const denox::io::Path &path = lit.as_path();

    // Database paths may not exist yet
    if (!path.exists()) {
      if (path.extension() == ".db") {
        return ArtefactParseResult::success(
            Artefact(DbArtefact{IOEndpoint(path)}));
      }

      return ArtefactParseResult::failure(ArtefactParseError::PathDoesNotExist);
    }

    auto file = denox::io::File::open(path, denox::io::File::OpenMode::Read);
    std::vector<std::byte> data(file.size());
    file.read_exact(data);

    return resolve_artefact(data, IOEndpoint(path));
  }
  }

  std::abort(); // unreachable
}
