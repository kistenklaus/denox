#include "./Db.hpp"
#include "db.h"
#include <dnx.h>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

denox::runtime::Db denox::runtime::Db::open(const std::string &path) {
  Db out;

  out.path = path;

  if (!std::filesystem::exists(path)) {
    return out;
  }

  std::ifstream dbFile(path, std::ios::binary);
  std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(dbFile)),
                                std::istreambuf_iterator<char>());

  flatbuffers::Verifier verifier(buf.data(), buf.size());

  if (!denox::db::VerifyDbBuffer(verifier)) {
    return out;
  }

  const denox::db::Db *db = denox::db::GetDb(buf.data());

  if (db->shader_binaries() != nullptr) {
    out.binaries.reserve(db->shader_binaries()->size());
    for (unsigned int i = 0; i < db->shader_binaries()->size(); ++i) {
      const denox::db::ShaderBinary *shaderBinary =
          db->shader_binaries()->Get(i);
      out.binaries.push_back(ShaderBinary{
          .source = {shaderBinary->spirv()->begin(),
                     shaderBinary->spirv()->end()},
      });
    }
  }

  if (db->dispatch() != nullptr) {
    out.dispatches.reserve(db->dispatch()->size());
    for (unsigned int i = 0; i < db->dispatch()->size(); ++i) {
      const denox::db::ComputeDispatch *dispatch = db->dispatch()->Get(i);

      std::vector<TensorBinding> bindings(
          dispatch->bindings()->size());
      for (unsigned int b = 0; b < dispatch->bindings()->size(); ++b) {
        const auto &bb = dispatch->bindings()->Get(b);
        bindings[b].set = bb->set();
        bindings[b].binding = bb->binding();
        switch (bb->access()) {
        case denox::db::Access_ReadOnly:
          bindings[b].access = Access::ReadOnly;
          break;
        case denox::db::Access_WriteOnly:
          bindings[b].access = Access::WriteOnly;
          break;
        case denox::db::Access_ReadWrite:
          bindings[b].access = Access::ReadWrite;
          break;
        }
        bindings[b].byteSize = bb->tensor_byte_size();
        bindings[b].minAlignment = bb->tensor_min_align();
      }

      Timing time{};
      if (dispatch->time() != nullptr) {
        time.samples = dispatch->time()->samples();
        time.latency_ns = dispatch->time()->latency_ns();
        time.std_derivation_ns = dispatch->time()->std_derivation_ns();
      }

      out.dispatches.push_back(ComputeDispatch{
          .binaryId = dispatch->binary_id(),
          .workgroupCountX = dispatch->workgroup_count_x(),
          .workgroupCountY = dispatch->workgroup_count_y(),
          .workgroupCountZ = dispatch->workgroup_count_z(),
          .pushConstant =
              dispatch->push_constant() == nullptr
                  ? (std::vector<uint8_t>())
                  : std::vector<uint8_t>(dispatch->push_constant()->begin(),
                                         dispatch->push_constant()->end()),
          .bindings = std::move(bindings),
          .time = time,
      });
    }
  }

  if (db->operations() != nullptr) {
    out.operations.reserve(db->operations()->size());
    for (unsigned int i = 0; i < db->operations()->size(); ++i) {
      const auto &op = db->operations()->Get(i);
      out.operations.push_back(Operation{
          .shaderName = op->shader_name()->str(),
          .pattern = op->pattern(),
          .config = op->config(),
          .hash = op->hash(),
          .dispatches = {op->dispatches()->begin(), op->dispatches()->end()}});
    }
  }

  return out;
}

void denox::runtime::Db::write_back() {
  if (path.empty()) {
    return;
  }

  std::string tmpPath = path + ".tmp";

  flatbuffers::FlatBufferBuilder fbb(1 << 16);

  std::vector<flatbuffers::Offset<denox::db::ShaderBinary>> binaries;

  for (const auto &binary : this->binaries) {
    binaries.push_back(denox::db::CreateShaderBinary(
        fbb, fbb.CreateVector<uint32_t>(binary.source.data(),
                                        binary.source.size())));
  }

  auto binariesVec = fbb.CreateVector(binaries);

  std::vector<flatbuffers::Offset<denox::db::ComputeDispatch>> dispatches;

  for (const auto &dispatch : this->dispatches) {

    std::vector<flatbuffers::Offset<denox::db::TensorBinding>> bindings;
    for (const auto &binding : dispatch.bindings) {
      denox::db::Access access;
      switch (binding.access) {
      case Access::ReadOnly:
        access = denox::db::Access_ReadOnly;
        break;
      case Access::WriteOnly:
        access = denox::db::Access_WriteOnly;
        break;
      case Access::ReadWrite:
        access = denox::db::Access_ReadWrite;
        break;
      }
      bindings.push_back(denox::db::CreateTensorBinding(
          fbb, binding.set, binding.binding, access, binding.byteSize,
          binding.minAlignment));
    }

    flatbuffers::Offset<denox::db::Timing> timing = denox::db::CreateTiming(
        fbb, dispatch.time.samples, dispatch.time.latency_ns,
        dispatch.time.std_derivation_ns);

    dispatches.push_back(denox::db::CreateComputeDispatch(
        fbb, dispatch.binaryId, dispatch.workgroupCountX,
        dispatch.workgroupCountY, dispatch.workgroupCountZ,
        fbb.CreateVector<uint8_t>(dispatch.pushConstant),
        fbb.CreateVector(bindings), timing));
  }

  auto dispatchesVec = fbb.CreateVector(dispatches);

  std::vector<flatbuffers::Offset<denox::db::Operation>> operations;
  for (const auto &op : this->operations) {
    operations.push_back(denox::db::CreateOperation(
        fbb, fbb.CreateString(op.shaderName), op.pattern, op.config, op.hash,
        fbb.CreateVector<uint32_t>(op.dispatches.data(),
                                   op.dispatches.size())));
  }

  auto operationsVec = fbb.CreateVector(operations);

  denox::db::DbBuilder builder(fbb);
  builder.add_shader_binaries(binariesVec);
  builder.add_dispatch(dispatchesVec);
  builder.add_operations(operationsVec);

  auto db = builder.Finish();
  denox::db::FinishDbBuffer(fbb, db);
  flatbuffers::DetachedBuffer detachedBuffer = fbb.Release();
  flatbuffers::Verifier v(detachedBuffer.data(), detachedBuffer.size());
  if (!denox::db::VerifyDbBuffer(v)) {
    throw std::runtime_error("Invalid db file.");
  }

  std::fstream fstream = std::fstream(
      tmpPath, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!fstream.is_open()) {
    fmt::println("\x1B[31m[Error:]\x1B[0m Failed to write db to \"{}\"",
                 tmpPath);
  } else {
    fstream.write(reinterpret_cast<const char *>(detachedBuffer.data()),
                  static_cast<long long>(detachedBuffer.size()));
    fstream.close();
    std::filesystem::rename(tmpPath, path);
  }
}
