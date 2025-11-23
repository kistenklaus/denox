#include "db/Db.hpp"
#include "db.h"
#include "diag/invalid_state.hpp"
#include "memory/container/vector.hpp"
#include <dnx.h>
#include <filesystem>
#include <fstream>
#include <iostream>

denox::compiler::Db denox::compiler::Db::open(const io::Path &path) {
  auto out = Db{std::make_shared<db::details::Db>()};
  db::details::Db *con = out.m_db.get();

  con->path = path;

  if (!path.exists()) {
    return out;
  }
  std::string spath = path.str();
  std::ifstream dbFile(spath.c_str(), std::ios::binary);
  std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(dbFile)),
                                std::istreambuf_iterator<char>());

  flatbuffers::Verifier verifier(buf.data(), buf.size());

  if (!denox::db::VerifyDbBuffer(verifier)) {
    return out;
  }

  const denox::db::Db *db = denox::db::GetDb(buf.data());

  if (db->shader_binaries() != nullptr) {
    con->binaries.reserve(db->shader_binaries()->size());
    for (unsigned int i = 0; i < db->shader_binaries()->size(); ++i) {
      const denox::db::ShaderBinary *shaderBinary =
          db->shader_binaries()->Get(i);
      con->binaries.push_back(db::details::ShaderBinary{
          .source = {shaderBinary->spirv()->begin(),
                     shaderBinary->spirv()->end()},
      });
    }
  }

  if (db->dispatch() != nullptr) {
    con->dispatches.reserve(db->dispatch()->size());
    for (unsigned int i = 0; i < db->dispatch()->size(); ++i) {
      const denox::db::ComputeDispatch *dispatch = db->dispatch()->Get(i);

      std::vector<db::details::TensorBinding> bindings(
          dispatch->bindings()->size());
      for (unsigned int b = 0; b < dispatch->bindings()->size(); ++b) {
        const auto &bb = dispatch->bindings()->Get(b);
        bindings[b].set = bb->set();
        bindings[b].binding = bb->binding();
        switch (bb->access()) {
        case denox::db::Access_ReadOnly:
          bindings[b].access = db::details::Access::ReadOnly;
          break;
        case denox::db::Access_WriteOnly:
          bindings[b].access = db::details::Access::WriteOnly;
          break;
        case denox::db::Access_ReadWrite:
          bindings[b].access = db::details::Access::ReadWrite;
          break;
        }
        bindings[b].byteSize = bb->tensor_byte_size();
        bindings[b].minAlignment = bb->tensor_min_align();
      }

      db::details::Timing time{};
      if (dispatch->time() != nullptr) {
        time.samples = dispatch->time()->samples();
        time.latency_ns = dispatch->time()->latency_ns();
        time.std_derivation_ns = dispatch->time()->std_derivation_ns();
      }

      con->dispatches.push_back(db::details::ComputeDispatch{
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
    con->operations.reserve(db->operations()->size());
    for (unsigned int i = 0; i < db->operations()->size(); ++i) {
      const auto &op = db->operations()->Get(i);
      con->operations.push_back(db::details::Operation{
          .shaderName = op->shader_name()->str(),
          .pattern = op->pattern(),
          .config = op->config(),
          .hash = op->hash(),
          .dispatches = {op->dispatches()->begin(), op->dispatches()->end()}});
    }
  }

  return out;
}

denox::compiler::db::details::Db::~Db() { close(); }

void denox::compiler::db::details::Db::close() {
  if (path.empty()) {
    return;
  }

  io::Path tmpPath = io::Path(path.str() + ".tmp").normalized();

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

    denox::db::CreateComputeDispatch(
        fbb, dispatch.binaryId, dispatch.workgroupCountX,
        dispatch.workgroupCountY, dispatch.workgroupCountZ,
        fbb.CreateVector<uint8_t>(dispatch.pushConstant),
        fbb.CreateVector(bindings), timing);
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
    diag::invalid_state();
  }

  std::fstream fstream = std::fstream(
      tmpPath.str(), std::ios::out | std::ios::binary | std::ios::trunc);
  if (!fstream.is_open()) {
    fmt::println("\x1B[31m[Error:]\x1B[0m Failed to write db to \"{}\"",
                 tmpPath.str());
  } else {
    fstream.write(reinterpret_cast<const char *>(detachedBuffer.data()),
                  static_cast<long long>(detachedBuffer.size()));
    fstream.close();
    std::filesystem::rename(tmpPath.str(), path.str());
  }
  path = io::Path();
}

uint64_t denox::compiler::db::details::Db::addShaderBinary(
    const denox::compiler::ShaderBinary &shaderBinary) {
  for (uint64_t b = 0; b < binaries.size(); ++b) {
    bool matches = binaries[b].source.size() == shaderBinary.spv.size();
    for (size_t i = 0; i < shaderBinary.spv.size() && matches; ++i) {
      matches = shaderBinary.spv[i] == binaries[b].source[i];
    }
    if (matches) {
      return b;
    }
  }
  uint64_t b = binaries.size();
  binaries.push_back(db::details::ShaderBinary{
      .source = shaderBinary.spv,
  });

  return b;
}
uint64_t
denox::compiler::Db::addShaderBinary(const ShaderBinary &shaderBinary) {
  assert(m_db != nullptr);
  return m_db->addShaderBinary(shaderBinary);
}

uint64_t denox::compiler::db::details::Db::addComputeDispatch(
    const denox::compiler::DbComputeDispatch &dispatch) {
  for (uint64_t d = 0; d < dispatches.size(); ++d) {
    if (dispatches[d].binaryId != dispatch.binaryId) {
      continue;
    }
    if (dispatches[d].workgroupCountX != dispatch.workgroupCountX) {
      continue;
    }
    if (dispatches[d].workgroupCountY != dispatch.workgroupCountY) {
      continue;
    }
    if (dispatches[d].workgroupCountZ != dispatch.workgroupCountZ) {
      continue;
    }
    bool matches = dispatch.bindings.size() == dispatches[d].bindings.size();
    for (size_t i = 0; i < dispatch.bindings.size() && matches; ++i) {
      const auto &lhs = dispatch.bindings[i];
      const auto &rhs = dispatches[d].bindings[i];
      if (static_cast<uint64_t>(lhs.access) !=
          static_cast<uint64_t>(rhs.access)) {
        matches = false;
      }
      if (lhs.binding != rhs.binding) {
        matches = false;
      }
      if (lhs.set != rhs.set) {
        matches = false;
      }
      if (lhs.tensorByteSize != rhs.byteSize) {
        matches = false;
      }
      if (lhs.tensorMinAlignment != rhs.minAlignment) {
        matches = false;
      }
    }
    if (!matches) {
      continue;
    }
    matches = dispatch.pushConstant.size() == dispatches[d].pushConstant.size();
    for (size_t i = 0; i < dispatch.pushConstant.size() && matches; ++i) {
      const auto &lhs = dispatch.pushConstant[i];
      const auto &rhs = dispatches[d].pushConstant[i];
      if (lhs != rhs) {
        matches = false;
      }
    }
    if (matches) {
      return d;
    }
  }
  std::vector<db::details::TensorBinding> bindings;
  for (unsigned int b = 0; b < dispatch.bindings.size(); ++b) {
    const auto &bb = dispatch.bindings[b];
    db::details::Access access;
    switch (bb.access) {
    case AccessFlag::ReadOnly:
      access = Access::ReadOnly;
      break;
    case AccessFlag::WriteOnly:
      access = Access::WriteOnly;
      break;
    case AccessFlag::ReadWrite:
      access = Access::ReadWrite;
      break;
    }

    bindings.push_back(db::details::TensorBinding{
        .set = bb.set,
        .binding = bb.binding,
        .access = access,
        .byteSize = bb.tensorByteSize,
        .minAlignment = bb.tensorMinAlignment,
    });
  }
  uint64_t d = dispatches.size();
  dispatches.push_back(db::details::ComputeDispatch{
      .binaryId = dispatch.binaryId,
      .workgroupCountX = dispatch.workgroupCountX,
      .workgroupCountY = dispatch.workgroupCountY,
      .workgroupCountZ = dispatch.workgroupCountZ,
      .pushConstant = dispatch.pushConstant,
      .bindings = bindings,
      .time = {},
  });
  return d;
}

uint64_t denox::compiler::Db::addComputeDispatch(
    const denox::compiler::DbComputeDispatch &dispatch) {
  assert(m_db != nullptr);
  return m_db->addComputeDispatch(dispatch);
}

uint64_t
denox::compiler::db::details::Db::addOp(const denox::compiler::DbOp &op) {
  for (uint64_t o = 0; o < operations.size(); ++o) {
    const auto &rhs = operations[o];
    if (rhs.shaderName != op.shaderName) {
      continue;
    }
    if (rhs.pattern != op.pattern) {
      continue;
    }
    if (rhs.config != op.config) {
      continue;
    }
    if (rhs.hash != op.hash) {
      continue;
    }
    bool matches = rhs.dispatches.size() == op.dispatches.size();
    for (size_t i = 0; i < op.dispatches.size() && matches; ++i) {
      matches = op.dispatches[i] != op.dispatches[i];
    }
    if (matches) {
      return o;
    }
  }

  uint64_t o = operations.size();
  operations.push_back(db::details::Operation{
      .shaderName = op.shaderName,
      .pattern = op.pattern,
      .config = op.config,
      .hash = op.hash,
      .dispatches = op.dispatches,
  });
  return o;
}

uint64_t denox::compiler::Db::addOp(const denox::compiler::DbOp &op) {
  assert(m_db != nullptr);
  return m_db->addOp(op);
}
