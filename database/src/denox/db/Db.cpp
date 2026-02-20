#include "denox/db/Db.hpp"

namespace denox {

static constexpr int DB_VERSION = 1;

Db Db::open(const io::Path &path) {
  sqlite::Db db = sqlite::Db::open(path);
  int version = 1;
  {
    auto stmt = db.prepare("PRAGMA user_version;");
    if (!stmt.next()) {
      throw std::runtime_error("Failed to read user_version");
    }
    version = stmt.as_int(0);
  }
  if (version != DB_VERSION) {
    db.with_transaction([&] {
      // ---- drop tables ----
      db.exec("DROP TABLE IF EXISTS timing_samples;");
      db.exec("DROP TABLE IF EXISTS dispatch_bindings;");
      db.exec("DROP TABLE IF EXISTS dispatches;");
      db.exec("DROP TABLE IF EXISTS shader_binaries;");
      db.exec("DROP TABLE IF EXISTS envs;");
      db.exec("CREATE TABLE envs ("
              "  id INTEGER PRIMARY KEY,"
              "  device TEXT NOT NULL,"
              "  os TEXT NOT NULL,"
              "  driver_version TEXT NOT NULL,"
              "  denox_version TEXT NOT NULL,"
              "  denox_commit_hash TEXT NOT NULL,"
              "  start_timestamp INTEGER NOT NULL,"
              "  clock_mode INTEGER NOT NULL,"
              "  l2_warmup_iterations INTEGER NOT NULL,"
              "  jit_warmup_iterations INTEGER NOT NULL,"
              "  measurement_iterations INTEGER NOT NULL"
              ");");
      db.exec("CREATE TABLE shader_binaries ("
              "  id INTEGER PRIMARY KEY,"
              "  src_sha256 BLOB NOT NULL,"
              "  spirv BLOB NOT NULL"
              ");");
      db.exec("CREATE UNIQUE INDEX shader_binaries_sha_idx "
              "ON shader_binaries(src_sha256);");
      db.exec("CREATE TABLE dispatches ("
              "  id INTEGER PRIMARY KEY,"
              "  binary_id INTEGER NOT NULL,"
              "  wg_x INTEGER NOT NULL,"
              "  wg_y INTEGER NOT NULL,"
              "  wg_z INTEGER NOT NULL,"
              "  push_constant BLOB NOT NULL,"
              "  hash INTEGER NOT NULL,"
              "  operation TEXT,"
              "  shader_name TEXT,"
              "  config TEXT,"
              "  memory_reads INTEGER,"
              "  memory_writes INTEGER,"
              "  flops INTEGER,"
              "  coopmat INTEGER,"
              "  fixed_subgroup_size INTEGER,"
              "  input_bindings BLOB,"
              "  output_bindings BLOB,"
              "  mean_latency_ns INTEGER,"
              "  std_derivation_ns INTEGER,"
              ""
              "  FOREIGN KEY(binary_id)"
              "    REFERENCES shader_binaries(id)"
              "    ON DELETE CASCADE"
              ");");
      db.exec("CREATE INDEX dispatches_hash_idx "
              "ON dispatches(hash);");
      db.exec("CREATE TABLE dispatch_bindings ("
              "  dispatch_id INTEGER NOT NULL,"
              "  idx INTEGER NOT NULL,"
              "  set_ INTEGER NOT NULL,"
              "  binding INTEGER NOT NULL,"
              "  access INTEGER NOT NULL,"
              "  format INTEGER NOT NULL,"
              "  storage INTEGER NOT NULL,"
              "  byte_size INTEGER NOT NULL,"
              "  alignment INTEGER NOT NULL,"
              "  width INTEGER,"
              "  height INTEGER,"
              "  channels INTEGER,"
              "  dtype INTEGER,"
              "  is_param INTEGER NOT NULL,"
              ""
              "  PRIMARY KEY(dispatch_id, idx),"
              ""
              "  FOREIGN KEY(dispatch_id)"
              "    REFERENCES dispatches(id)"
              "    ON DELETE CASCADE"
              ");");
      db.exec("CREATE TABLE timing_samples ("
              "  dispatch_id INTEGER NOT NULL,"
              "  idx INTEGER NOT NULL,"
              "  timestamp INTEGER NOT NULL,"
              "  latency_ns INTEGER NOT NULL,"
              "  env INTEGER NOT NULL,"
              "  gpu_clock INTEGER,"
              "  mem_clock INTEGER,"
              ""
              "  PRIMARY KEY(dispatch_id, idx),"
              ""
              "  FOREIGN KEY(dispatch_id)"
              "    REFERENCES dispatches(id)"
              "    ON DELETE CASCADE,"
              ""
              "  FOREIGN KEY(env)"
              "    REFERENCES envs(id)"
              "    ON DELETE RESTRICT"
              ");");
      // ---- set version ----
      db.exec(fmt::format("PRAGMA user_version = {};", DB_VERSION));
    });
  }
  Inner *inner = new Inner{
      .db = std::move(db),
      .mutex = {},
      .query_binary_by_hash = {},
      .query_dispatch_latency = {},
      .query_binary_existence = {},
      .insert_binary = {},
      .insert_dispatch_query_dispatch_existance = {},
      .insert_dispatch_insert_dispatch = {},
      .insert_dispatch_insert_bindings = {},
  };

  auto out = Db(std::shared_ptr<Inner>(inner));
  out.create_cached_stmts();
  return out;
}

void Db::checkpoint() {
  std::lock_guard lck{m_inner->mutex};
  finalize_stmts();
  m_inner->db.checkpoint();
  create_cached_stmts();
}

std::optional<SpirvBinary>
Db::query_shader_binary(const SHA256 &srcHash) const {
  std::lock_guard lck{m_inner->mutex};
  auto &stmt = m_inner->query_binary_by_hash;
  stmt.reset();
  stmt.clear_bindings();
  stmt.bind_blob(1, srcHash.h, static_cast<int>(sizeof(uint32_t) * 8));
  if (!stmt.next()) {
    return std::nullopt;
  }
  auto blob = stmt.as_blob_view(0);
  if (blob.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("Invalid SPIR-V blob size in database");
  }
  size_t word_count = blob.size() / sizeof(uint32_t);
  SpirvBinary out;
  out.spv.resize(word_count);
  std::memcpy(out.spv.data(), blob.data(), blob.size());
  return out;
}

std::optional<std::chrono::duration<float, std::milli>>
Db::query_dispatch_latency(const SHA256 &srcHash,
                           std::span<const uint8_t> pushConstant,
                           uint32_t workgroupCountX, uint32_t workgroupCountY,
                           uint32_t workgroupCountZ) const {
  std::lock_guard lck{m_inner->mutex};
  uint64_t hash = std::hash<SHA256>{}(srcHash);
  for (uint8_t b : pushConstant) {
    hash = algorithm::hash_combine(hash, b);
  }
  hash = algorithm::hash_combine(hash, workgroupCountX);
  hash = algorithm::hash_combine(hash, workgroupCountY);
  hash = algorithm::hash_combine(hash, workgroupCountZ);
  auto &stmt = m_inner->query_dispatch_latency;
  stmt.reset();
  stmt.clear_bindings();
  stmt.bind_int64(1, static_cast<int64_t>(hash));
  stmt.bind_int(2, static_cast<int>(workgroupCountX));
  stmt.bind_int(3, static_cast<int>(workgroupCountY));
  stmt.bind_int(4, static_cast<int>(workgroupCountZ));
  stmt.bind_blob(5, pushConstant.data(), static_cast<int>(pushConstant.size()));
  stmt.bind_blob(6, srcHash.h, static_cast<int>(sizeof(uint32_t) * 8));
  if (!stmt.next()) {
    return std::nullopt;
  }
  uint64_t mean_ns = static_cast<uint64_t>(stmt.as_int64(0));
  std::chrono::duration<uint64_t, std::nano> ns(mean_ns);
  return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
      ns);
}

bool Db::insert_dispatch(
    const SHA256 &srcHash, std::span<const uint8_t> pushConstant,
    uint32_t workgroupCountX, uint32_t workgroupCountY,
    uint32_t workgroupCountZ, std::span<const DbTensorBinding> bindings,
    const SpirvBinary &spvBinary, memory::optional<memory::string> operation,
    memory::optional<memory::string> shader_name,
    memory::optional<memory::string> config,
    memory::optional<uint64_t> memory_reads,
    memory::optional<uint64_t> memory_writes, memory::optional<uint64_t> flops,
    memory::optional<bool> coopmat,
    memory::optional<std::span<const uint32_t>> input_bindings,
    memory::optional<std::span<const uint32_t>> output_bindings,
    memory::optional<uint32_t> subgroupSize) {
  std::lock_guard lck{m_inner->mutex};
  return m_inner->db.with_transaction([&]() -> bool {
    uint64_t binaryId = 0;
    {
      auto &stmt = m_inner->query_binary_existence;
      stmt.reset();
      stmt.clear_bindings();
      stmt.bind_blob(1, srcHash.h, sizeof(uint32_t) * 8);
      if (stmt.next()) {
        binaryId = static_cast<uint64_t>(stmt.as_int64(0));
        auto blob = stmt.as_blob_view(1);
        if (blob.size() != spvBinary.spv.size() * sizeof(uint32_t) ||
            std::memcmp(blob.data(), spvBinary.spv.data(), blob.size()) != 0) {
          throw std::runtime_error("SPIR-V mismatch for identical shader hash");
        }
      } else {
        auto &ins = m_inner->insert_binary;
        ins.reset();
        ins.clear_bindings();
        ins.bind_blob(1, srcHash.h, sizeof(uint32_t) * 8);
        ins.bind_blob(
            2, spvBinary.spv.data(),
            static_cast<int>(spvBinary.spv.size() * sizeof(uint32_t)));
        ins.next();
        binaryId = static_cast<uint64_t>(m_inner->db.last_insert_rowid());
      }
    }
    uint64_t hash = std::hash<SHA256>{}(srcHash);
    for (uint8_t b : pushConstant) {
      hash = algorithm::hash_combine(hash, b);
    }
    hash = algorithm::hash_combine(hash, workgroupCountX);
    hash = algorithm::hash_combine(hash, workgroupCountY);
    hash = algorithm::hash_combine(hash, workgroupCountZ);
    {
      auto &stmt = m_inner->insert_dispatch_query_dispatch_existance;
      stmt.reset();
      stmt.clear_bindings();
      stmt.bind_int64(1, static_cast<int64_t>(hash));
      stmt.bind_int64(2, static_cast<int64_t>(binaryId));
      stmt.bind_int64(3, workgroupCountX);
      stmt.bind_int64(4, workgroupCountY);
      stmt.bind_int64(5, workgroupCountZ);
      stmt.bind_blob(6, pushConstant.data(),
                     static_cast<int>(pushConstant.size()));
      if (input_bindings) {
        stmt.bind_blob(
            7, input_bindings->data(),
            static_cast<int>(input_bindings->size() * sizeof(uint32_t)));
      } else {
        stmt.bind_null(7);
      }
      if (output_bindings) {
        stmt.bind_blob(
            8, output_bindings->data(),
            static_cast<int>(output_bindings->size() * sizeof(uint32_t)));
      } else {
        stmt.bind_null(8);
      }
      if (stmt.next()) {
        return false;
      }
    }
    uint64_t dispatchId = 0;
    {
      auto &ins = m_inner->insert_dispatch_insert_dispatch;
      ins.reset();
      ins.clear_bindings();
      ins.bind_int64(1, static_cast<int64_t>(binaryId));
      ins.bind_int64(2, workgroupCountX);
      ins.bind_int64(3, workgroupCountY);
      ins.bind_int64(4, workgroupCountZ);
      ins.bind_blob(5, pushConstant.data(),
                    static_cast<int>(pushConstant.size()));
      ins.bind_int64(6, static_cast<int64_t>(hash));
      operation ? ins.bind_sv(7, *operation) : ins.bind_null(7);
      shader_name ? ins.bind_sv(8, *shader_name) : ins.bind_null(8);
      config ? ins.bind_sv(9, *config) : ins.bind_null(9);
      memory_reads ? ins.bind_int64(10, static_cast<int64_t>(*memory_reads))
                   : ins.bind_null(10);
      memory_writes ? ins.bind_int64(11, static_cast<int64_t>(*memory_writes))
                    : ins.bind_null(11);
      flops ? ins.bind_int64(12, static_cast<int64_t>(*flops))
            : ins.bind_null(12);
      coopmat ? ins.bind_int(13, *coopmat ? 1 : 0) : ins.bind_null(13);
      subgroupSize ? ins.bind_int64(14, static_cast<int64_t>(*subgroupSize))
                   : ins.bind_null(14);
      // ---- input_bindings ----
      if (input_bindings) {
        ins.bind_blob(
            15, input_bindings->data(),
            static_cast<int>(input_bindings->size() * sizeof(uint32_t)));
      } else {
        ins.bind_null(15);
      }
      // ---- output_bindings ----
      if (output_bindings) {
        ins.bind_blob(
            16, output_bindings->data(),
            static_cast<int>(output_bindings->size() * sizeof(uint32_t)));
      } else {
        ins.bind_null(16);
      }
      ins.next();
      dispatchId = static_cast<uint64_t>(m_inner->db.last_insert_rowid());
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto &b = bindings[i];
      auto &ins = m_inner->insert_dispatch_insert_bindings;
      ins.reset();
      ins.clear_bindings();
      ins.bind_int64(1, static_cast<int64_t>(dispatchId));
      ins.bind_int64(2, static_cast<int64_t>(i));
      ins.bind_int64(3, b.set);
      ins.bind_int64(4, b.binding);
      ins.bind_int64(5, static_cast<int>(b.access));
      ins.bind_int64(6, static_cast<int>(b.format));
      ins.bind_int64(7, static_cast<int>(b.storage));
      ins.bind_int64(8, static_cast<int64_t>(b.byteSize));
      ins.bind_int(9, b.alignment);
      b.width ? ins.bind_int64(10, *b.width) : ins.bind_null(10);
      b.height ? ins.bind_int64(11, *b.height) : ins.bind_null(11);
      b.channels ? ins.bind_int64(12, *b.channels) : ins.bind_null(12);
      b.type ? ins.bind_int64(13, static_cast<int>(*b.type))
             : ins.bind_null(13);
      ins.bind_int(14, b.is_param ? 1 : 0);
      ins.next();
    }
    return true;
  });
}

bool Db::insert_dispatch(
    const SHA256 &srcHash, std::span<const uint8_t> pushConstant,
    uint32_t workgroupCountX, uint32_t workgroupCountY,
    uint32_t workgroupCountZ, std::span<const DbTensorBinding> bindings,
    uint32_t binary_id, memory::optional<memory::string> operation,
    memory::optional<memory::string> shader_name,
    memory::optional<memory::string> config,
    memory::optional<uint64_t> memory_reads,
    memory::optional<uint64_t> memory_writes, memory::optional<uint64_t> flops,
    memory::optional<bool> coopmat,
    memory::optional<std::span<const uint32_t>> input_bindings,
    memory::optional<std::span<const uint32_t>> output_bindings,
    memory::optional<uint32_t> subgroupSize) {
  std::lock_guard lck{m_inner->mutex};
  return m_inner->db.with_transaction([&]() -> bool {
    uint64_t hash = std::hash<SHA256>{}(srcHash);
    for (uint8_t b : pushConstant) {
      hash = algorithm::hash_combine(hash, b);
    }
    hash = algorithm::hash_combine(hash, workgroupCountX);
    hash = algorithm::hash_combine(hash, workgroupCountY);
    hash = algorithm::hash_combine(hash, workgroupCountZ);
    {
      auto &stmt = m_inner->insert_dispatch_query_dispatch_existance;
      stmt.reset();
      stmt.clear_bindings();
      stmt.bind_int64(1, static_cast<int64_t>(hash));
      stmt.bind_int64(2, static_cast<int64_t>(binary_id));
      stmt.bind_int64(3, workgroupCountX);
      stmt.bind_int64(4, workgroupCountY);
      stmt.bind_int64(5, workgroupCountZ);
      stmt.bind_blob(6, pushConstant.data(),
                     static_cast<int>(pushConstant.size()));
      if (input_bindings) {
        stmt.bind_blob(
            7, input_bindings->data(),
            static_cast<int>(input_bindings->size() * sizeof(uint32_t)));
      } else {
        stmt.bind_null(7);
      }
      if (output_bindings) {
        stmt.bind_blob(
            8, output_bindings->data(),
            static_cast<int>(output_bindings->size() * sizeof(uint32_t)));
      } else {
        stmt.bind_null(8);
      }
      if (stmt.next()) {
        return false;
      }
    }
    uint64_t dispatchId = 0;
    {
      auto &ins = m_inner->insert_dispatch_insert_dispatch;
      ins.reset();
      ins.clear_bindings();
      ins.bind_int64(1, static_cast<int64_t>(binary_id));
      ins.bind_int64(2, workgroupCountX);
      ins.bind_int64(3, workgroupCountY);
      ins.bind_int64(4, workgroupCountZ);
      ins.bind_blob(5, pushConstant.data(),
                    static_cast<int>(pushConstant.size()));
      ins.bind_int64(6, static_cast<int64_t>(hash));
      operation ? ins.bind_sv(7, *operation) : ins.bind_null(7);
      shader_name ? ins.bind_sv(8, *shader_name) : ins.bind_null(8);
      config ? ins.bind_sv(9, *config) : ins.bind_null(9);
      memory_reads ? ins.bind_int64(10, static_cast<int64_t>(*memory_reads))
                   : ins.bind_null(10);
      memory_writes ? ins.bind_int64(11, static_cast<int64_t>(*memory_writes))
                    : ins.bind_null(11);
      flops ? ins.bind_int64(12, static_cast<int64_t>(*flops))
            : ins.bind_null(12);
      coopmat ? ins.bind_int(13, *coopmat ? 1 : 0) : ins.bind_null(13);
      subgroupSize ? ins.bind_int64(14, static_cast<int64_t>(*subgroupSize))
                   : ins.bind_null(14);
      // ---- input_bindings ----
      if (input_bindings) {
        ins.bind_blob(
            15, input_bindings->data(),
            static_cast<int>(input_bindings->size() * sizeof(uint32_t)));
      } else {
        ins.bind_null(15);
      }
      // ---- output_bindings ----
      if (output_bindings) {
        ins.bind_blob(
            16, output_bindings->data(),
            static_cast<int>(output_bindings->size() * sizeof(uint32_t)));
      } else {
        ins.bind_null(16);
      }
      ins.next();
      dispatchId = static_cast<uint64_t>(m_inner->db.last_insert_rowid());
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto &b = bindings[i];
      auto &ins = m_inner->insert_dispatch_insert_bindings;
      ins.reset();
      ins.clear_bindings();
      ins.bind_int64(1, static_cast<int64_t>(dispatchId));
      ins.bind_int64(2, static_cast<int64_t>(i));
      ins.bind_int64(3, b.set);
      ins.bind_int64(4, b.binding);
      ins.bind_int64(5, static_cast<int>(b.access));
      ins.bind_int64(6, static_cast<int>(b.format));
      ins.bind_int64(7, static_cast<int>(b.storage));
      ins.bind_int64(8, static_cast<int64_t>(b.byteSize));
      ins.bind_int(9, b.alignment);
      b.width ? ins.bind_int64(10, *b.width) : ins.bind_null(10);
      b.height ? ins.bind_int64(11, *b.height) : ins.bind_null(11);
      b.channels ? ins.bind_int64(12, *b.channels) : ins.bind_null(12);
      b.type ? ins.bind_int64(13, static_cast<int>(*b.type))
             : ins.bind_null(13);
      ins.bind_int(14, b.is_param ? 1 : 0);
      ins.next();
    }
    return true;
  });
}

bool denox::Db::insert_binary(const SHA256 &srcHash,
                              const SpirvBinary &binary) {
  std::lock_guard lck{m_inner->mutex};
  return m_inner->db.with_transaction([&]() -> bool {
    auto &stmt = m_inner->query_binary_existence;
    stmt.reset();
    stmt.clear_bindings();
    stmt.bind_blob(1, srcHash.h, sizeof(uint32_t) * 8);
    if (stmt.next()) {
      auto blob = stmt.as_blob_view(1);
      if (blob.size() != binary.spv.size() * sizeof(uint32_t) ||
          std::memcmp(blob.data(), binary.spv.data(), blob.size()) != 0) {
        throw std::runtime_error(
            "Failed to insert binary into database: SHA256 collision.");
      }
      return false;
    }
    auto &ins = m_inner->insert_binary;
    ins.reset();
    ins.clear_bindings();
    ins.bind_blob(1, srcHash.h, sizeof(uint32_t) * 8);
    ins.bind_blob(2, binary.spv.data(),
                  static_cast<int>(binary.spv.size() * sizeof(uint32_t)));
    ins.next();
    return true;
  });
}

DbShaderBinary Db::queryShaderBinaryById(uint32_t id) const {
  std::lock_guard lck{m_inner->mutex};
  auto stmt = m_inner->db.prepare("SELECT src_sha256, spirv "
                                  "FROM shader_binaries "
                                  "WHERE id = ?1;");
  stmt.bind_int64(1, static_cast<int64_t>(id));
  if (!stmt.next()) {
    throw std::runtime_error("Shader binary not found");
  }
  DbShaderBinary out;
  auto hash_blob = stmt.as_blob_view(0);
  if (hash_blob.size() != sizeof(uint32_t) * 8) {
    throw std::runtime_error("Invalid SHA256 size in DB");
  }
  std::memcpy(out.hash.h, hash_blob.data(), hash_blob.size());
  auto spv_blob = stmt.as_blob_view(1);
  if (spv_blob.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("Invalid SPIR-V blob size");
  }
  size_t word_count = spv_blob.size() / sizeof(uint32_t);
  out.spvBinary.spv.resize(word_count);
  std::memcpy(out.spvBinary.spv.data(), spv_blob.data(), spv_blob.size());
  return out;
}

DbComputeDispatch Db::queryComputeDispatchById(uint32_t id) const {
  std::lock_guard lck{m_inner->mutex};
  auto stmt = m_inner->db.prepare("SELECT binary_id, wg_x, wg_y, wg_z, "
                                  "push_constant, hash, "
                                  "operation, shader_name, config, "
                                  "memory_reads, memory_writes, flops, "
                                  "coopmat, fixed_subgroup_size, "
                                  "input_bindings, output_bindings, "
                                  "mean_latency_ns, std_derivation_ns "
                                  "FROM dispatches "
                                  "WHERE id = ?1;");
  stmt.bind_int64(1, static_cast<int64_t>(id));
  if (!stmt.next()) {
    throw std::runtime_error("Dispatch not found");
  }
  DbComputeDispatch out{};
  out.binaryId = static_cast<uint32_t>(stmt.as_int64(0));
  out.workgroupCountX = static_cast<uint32_t>(stmt.as_int64(1));
  out.workgroupCountY = static_cast<uint32_t>(stmt.as_int64(2));
  out.workgroupCountZ = static_cast<uint32_t>(stmt.as_int64(3));
  {
    auto blob = stmt.as_blob_view(4);
    out.pushConstant.resize(blob.size());
    std::memcpy(out.pushConstant.data(), blob.data(), blob.size());
  }
  out.hash = static_cast<uint64_t>(stmt.as_int64(5));
  out.operation = stmt.as_optional_string(6);
  out.shader_name = stmt.as_optional_string(7);
  out.config = stmt.as_optional_string(8);
  out.memory_reads = stmt.as_optional_int64(9);
  out.memory_writes = stmt.as_optional_int64(10);
  out.flops = stmt.as_optional_int64(11);
  out.coopmat = stmt.as_optional_int(12).value_or(0) != 0;
  out.fixed_subgroup_size = stmt.as_optional_int64(13);
  if (!stmt.is_null(14)) {
    auto blob = stmt.as_blob_view(14);
    size_t count = blob.size() / sizeof(uint32_t);
    out.input_bindings.emplace(count);
    std::memcpy(out.input_bindings->data(), blob.data(), blob.size());
  }
  if (!stmt.is_null(15)) {
    auto blob = stmt.as_blob_view(15);
    size_t count = blob.size() / sizeof(uint32_t);
    out.output_bindings.emplace(count);
    std::memcpy(out.output_bindings->data(), blob.data(), blob.size());
  }
  if (!stmt.is_null(16)) { // mean_latency_ns not NULL
    DbDispatchTiming t{};
    t.mean_latency_ns = static_cast<uint64_t>(stmt.as_int64(16));
    t.std_derivation_ns = static_cast<uint64_t>(stmt.as_int64(17));
    auto count_stmt = m_inner->db.prepare(
        "SELECT COUNT(*) FROM timing_samples WHERE dispatch_id = ?1;");
    count_stmt.bind_int64(1, static_cast<int64_t>(id));
    if (count_stmt.next()) {
      uint64_t n = static_cast<uint64_t>(count_stmt.as_int64(0));
      t.samples.resize(n); // only size matters for convergence logic
    }
    out.time = std::move(t);
  }
  {
    auto bstmt = m_inner->db.prepare(
        "SELECT idx, set_, binding, access, format, storage, "
        "byte_size, alignment, width, height, channels, dtype, is_param "
        "FROM dispatch_bindings "
        "WHERE dispatch_id = ?1 "
        "ORDER BY idx ASC;");
    bstmt.bind_int64(1, static_cast<int64_t>(id));
    while (bstmt.next()) {
      DbTensorBinding b{};
      b.set = static_cast<uint32_t>(bstmt.as_int64(1));
      b.binding = static_cast<uint32_t>(bstmt.as_int64(2));
      b.access = static_cast<Access>(bstmt.as_int64(3));
      b.format = static_cast<TensorFormat>(bstmt.as_int64(4));
      b.storage = static_cast<TensorStorage>(bstmt.as_int64(5));
      b.byteSize = static_cast<uint64_t>(bstmt.as_int64(6));
      b.alignment = static_cast<uint16_t>(bstmt.as_int(7));
      b.width = bstmt.as_optional_int64(8);
      b.height = bstmt.as_optional_int64(9);
      b.channels = bstmt.as_optional_int64(10);
      if (!bstmt.is_null(11))
        b.type = static_cast<TensorDataType>(bstmt.as_int64(11));
      b.is_param = bstmt.as_int(12) != 0;
      out.bindings.push_back(std::move(b));
    }
  }
  return out;
}

memory::vector<DbComputeDispatch> Db::bulkQueryComputeDispatchById(
    memory::span<const uint32_t> dispatch_ids) const {
  std::lock_guard lck{m_inner->mutex};
  memory::vector<DbComputeDispatch> out;
  if (dispatch_ids.empty())
    return out;
  out.reserve(dispatch_ids.size());
  std::string in_clause = "(";
  for (size_t i = 0; i < dispatch_ids.size(); ++i) {
    if (i > 0)
      in_clause += ",";
    in_clause += "?";
  }
  in_clause += ")";
  {
    std::string sql = "SELECT id, binary_id, wg_x, wg_y, wg_z, "
                      "push_constant, hash, "
                      "operation, shader_name, config, "
                      "memory_reads, memory_writes, flops, "
                      "coopmat, fixed_subgroup_size, "
                      "input_bindings, output_bindings, "
                      "mean_latency_ns, std_derivation_ns "
                      "FROM dispatches "
                      "WHERE id IN " +
                      in_clause + ";";
    auto stmt = m_inner->db.prepare(sql.c_str());
    for (size_t i = 0; i < dispatch_ids.size(); ++i)
      stmt.bind_int64(static_cast<int>(i + 1),
                      static_cast<int64_t>(dispatch_ids[i]));
    memory::hash_map<uint32_t, size_t> id_to_index;
    id_to_index.reserve(dispatch_ids.size());
    while (stmt.next()) {
      DbComputeDispatch d{};
      const uint32_t id = static_cast<uint32_t>(stmt.as_int64(0));
      d.binaryId = static_cast<uint32_t>(stmt.as_int64(1));
      d.workgroupCountX = static_cast<uint32_t>(stmt.as_int64(2));
      d.workgroupCountY = static_cast<uint32_t>(stmt.as_int64(3));
      d.workgroupCountZ = static_cast<uint32_t>(stmt.as_int64(4));
      {
        auto blob = stmt.as_blob_view(5);
        d.pushConstant.resize(blob.size());
        std::memcpy(d.pushConstant.data(), blob.data(), blob.size());
      }
      d.hash = static_cast<uint64_t>(stmt.as_int64(6));
      d.operation = stmt.as_optional_string(7);
      d.shader_name = stmt.as_optional_string(8);
      d.config = stmt.as_optional_string(9);
      d.memory_reads = stmt.as_optional_int64(10);
      d.memory_writes = stmt.as_optional_int64(11);
      d.flops = stmt.as_optional_int64(12);
      d.coopmat = stmt.as_optional_int(13).value_or(0) != 0;
      d.fixed_subgroup_size = stmt.as_optional_int64(14);
      if (!stmt.is_null(15)) {
        auto blob = stmt.as_blob_view(15);
        size_t count = blob.size() / sizeof(uint32_t);
        d.input_bindings.emplace(count);
        std::memcpy(d.input_bindings->data(), blob.data(), blob.size());
      }
      if (!stmt.is_null(16)) {
        auto blob = stmt.as_blob_view(16);
        size_t count = blob.size() / sizeof(uint32_t);
        d.output_bindings.emplace(count);
        std::memcpy(d.output_bindings->data(), blob.data(), blob.size());
      }
      if (!stmt.is_null(17)) {
        DbDispatchTiming t{};
        t.mean_latency_ns = static_cast<uint64_t>(stmt.as_int64(17));
        t.std_derivation_ns = static_cast<uint64_t>(stmt.as_int64(18));
        d.time = std::move(t);
      }
      id_to_index[id] = out.size();
      out.push_back(std::move(d));
    }
    std::string sql2 =
        "SELECT dispatch_id, idx, set_, binding, access, format, "
        "storage, byte_size, alignment, width, height, channels, "
        "dtype, is_param "
        "FROM dispatch_bindings "
        "WHERE dispatch_id IN " +
        in_clause + " ORDER BY dispatch_id, idx ASC;";
    auto bstmt = m_inner->db.prepare(sql2.c_str());
    for (size_t i = 0; i < dispatch_ids.size(); ++i)
      bstmt.bind_int64(static_cast<int>(i + 1),
                       static_cast<int64_t>(dispatch_ids[i]));
    while (bstmt.next()) {
      uint32_t dispatch_id = static_cast<uint32_t>(bstmt.as_int64(0));
      auto it = id_to_index.find(dispatch_id);
      if (it == id_to_index.end())
        continue;
      auto &vec = out[it->second].bindings;
      DbTensorBinding b{};
      b.set = static_cast<uint32_t>(bstmt.as_int64(2));
      b.binding = static_cast<uint32_t>(bstmt.as_int64(3));
      b.access = static_cast<Access>(bstmt.as_int64(4));
      b.format = static_cast<TensorFormat>(bstmt.as_int64(5));
      b.storage = static_cast<TensorStorage>(bstmt.as_int64(6));
      b.byteSize = static_cast<uint64_t>(bstmt.as_int64(7));
      b.alignment = static_cast<uint16_t>(bstmt.as_int(8));
      b.width = bstmt.as_optional_int64(9);
      b.height = bstmt.as_optional_int64(10);
      b.channels = bstmt.as_optional_int64(11);
      if (!bstmt.is_null(12))
        b.type = static_cast<TensorDataType>(bstmt.as_int64(12));
      b.is_param = bstmt.as_int(13) != 0;
      vec.push_back(std::move(b));
    }
    std::string sql3 = "SELECT dispatch_id, COUNT(*) "
                       "FROM timing_samples "
                       "WHERE dispatch_id IN " +
                       in_clause + " GROUP BY dispatch_id;";
    auto tstmt = m_inner->db.prepare(sql3.c_str());
    for (size_t i = 0; i < dispatch_ids.size(); ++i)
      tstmt.bind_int64(static_cast<int>(i + 1),
                       static_cast<int64_t>(dispatch_ids[i]));
    while (tstmt.next()) {
      uint32_t dispatch_id = static_cast<uint32_t>(tstmt.as_int64(0));
      uint64_t count = static_cast<uint64_t>(tstmt.as_int64(1));
      auto it = id_to_index.find(dispatch_id);
      if (it == id_to_index.end())
        continue;
      if (out[it->second].time)
        out[it->second].time->samples.resize(count);
    }
  }
  return out;
}

DbEnv Db::queryEnvById(uint32_t id) const {
  std::lock_guard lck{m_inner->mutex};
  auto stmt =
      m_inner->db.prepare("SELECT device, os, driver_version, "
                          "denox_version, denox_commit_hash, "
                          "start_timestamp, clock_mode, "
                          "l2_warmup_iterations, jit_warmup_iterations, "
                          "measurement_iterations "
                          "FROM envs WHERE id = ?1;");
  stmt.bind_int64(1, static_cast<int64_t>(id));
  if (!stmt.next()) {
    throw std::runtime_error("Env not found");
  }
  DbEnv out{};
  out.device = stmt.as_string(0);
  out.os = stmt.as_string(1);
  out.driver_version = stmt.as_string(2);
  out.denox_version = stmt.as_string(3);
  out.denox_commit_hash = stmt.as_string(4);
  out.start_timestamp = static_cast<uint64_t>(stmt.as_int64(5));
  out.clock_mode = static_cast<DbClockMode>(stmt.as_int(6));
  out.l2_warmup_iterations = static_cast<uint16_t>(stmt.as_int(7));
  out.jit_warmup_iterations = static_cast<uint16_t>(stmt.as_int(8));
  out.measurement_iterations = static_cast<uint16_t>(stmt.as_int(9));
  return out;
}

uint32_t Db::queryComputeDispatchCount() const {
  std::lock_guard lck{m_inner->mutex};
  auto stmt = m_inner->db.prepare("SELECT COUNT(*) FROM dispatches;");
  if (!stmt.next()) {
    throw std::runtime_error("Failed to count dispatches");
  }
  return static_cast<uint32_t>(stmt.as_int64(0));
}

void Db::add_dispatch_benchmark_result(uint32_t dispatch_id,
                                       std::vector<DbSample> samples) {
  std::lock_guard lck{m_inner->mutex};
  if (samples.empty()) {
    return;
  }
  m_inner->db.with_transaction([&] {
    auto insert_sample = m_inner->db.prepare(
        "INSERT INTO timing_samples("
        "dispatch_id, idx, timestamp, latency_ns, env, gpu_clock, mem_clock"
        ") VALUES(?1,?2,?3,?4,?5,?6,?7);");
    uint64_t base_index = 0;
    {
      auto count_stmt =
          m_inner->db.prepare("SELECT COUNT(*) FROM timing_samples "
                              "WHERE dispatch_id = ?1;");
      count_stmt.bind_int64(1, dispatch_id);
      if (count_stmt.next())
        base_index = static_cast<uint64_t>(count_stmt.as_int64(0));
    }
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto &s = samples[i];
      insert_sample.reset();
      insert_sample.clear_bindings();
      insert_sample.bind_int64(1, dispatch_id);
      insert_sample.bind_int64(2, static_cast<int64_t>(base_index + i));
      insert_sample.bind_int64(3, static_cast<int64_t>(s.timestamp));
      insert_sample.bind_int64(4, static_cast<int64_t>(s.latency_ns));
      insert_sample.bind_int64(5, s.env);
      s.gpuClock ? insert_sample.bind_int64(6, s.gpuClock)
                 : insert_sample.bind_null(6);
      s.memClock ? insert_sample.bind_int64(7, s.memClock)
                 : insert_sample.bind_null(7);
      insert_sample.next();
    }
    auto stats_stmt = m_inner->db.prepare("SELECT COUNT(*), AVG(latency_ns), "
                                          "AVG(latency_ns * latency_ns) "
                                          "FROM timing_samples "
                                          "WHERE dispatch_id = ?1;");
    stats_stmt.bind_int64(1, dispatch_id);
    if (!stats_stmt.next()) {
      throw std::runtime_error("Failed to compute timing stats");
    }
    const double n = static_cast<double>(stats_stmt.as_int64(0));
    const double mean = stats_stmt.as_double(1);
    const double mean_sq = stats_stmt.as_double(2);
    if (n <= 0.0) {
      return;
    }
    const double variance = std::max(0.0, mean_sq - mean * mean);
    const double stddev = std::sqrt(variance);
    auto update_stmt = m_inner->db.prepare("UPDATE dispatches "
                                           "SET mean_latency_ns = ?1, "
                                           "    std_derivation_ns = ?2 "
                                           "WHERE id = ?3;");
    update_stmt.bind_int64(1, static_cast<int64_t>(mean));
    update_stmt.bind_int64(2, static_cast<int64_t>(stddev));
    update_stmt.bind_int64(3, dispatch_id);
    update_stmt.next();
  });
}

uint32_t denox::Db::create_bench_environment(
    std::string device, std::string os, std::string driver_version,
    std::string denox_version, std::string denox_commit_hash,
    uint64_t start_timestamp, DbClockMode clockMode,
    uint16_t l2_warmup_iterations, uint16_t jit_warmup_iterations,
    uint16_t measurement_iterations) {
  std::lock_guard lck{m_inner->mutex};
  auto &db = m_inner->db;
  auto stmt = db.prepare(
      "INSERT INTO envs("
      "device, os, driver_version, denox_version, "
      "denox_commit_hash, start_timestamp, clock_mode, "
      "l2_warmup_iterations, jit_warmup_iterations, measurement_iterations"
      ") VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10);");
  stmt.bind_sv(1, device);
  stmt.bind_sv(2, os);
  stmt.bind_sv(3, driver_version);
  stmt.bind_sv(4, denox_version);
  stmt.bind_sv(5, denox_commit_hash);
  stmt.bind_int64(6, static_cast<int64_t>(start_timestamp));
  stmt.bind_int(7, static_cast<int>(clockMode));
  stmt.bind_int(8, static_cast<int>(l2_warmup_iterations));
  stmt.bind_int(9, static_cast<int>(jit_warmup_iterations));
  stmt.bind_int(10, static_cast<int>(measurement_iterations));
  stmt.next();
  return static_cast<uint32_t>(db.last_insert_rowid());
}

void Db::finalize_stmts() {
  m_inner->query_binary_by_hash.finalize();
  m_inner->query_dispatch_latency.finalize();
  m_inner->query_binary_existence.finalize();
  m_inner->insert_binary.finalize();
  m_inner->insert_dispatch_query_dispatch_existance.finalize();
  m_inner->insert_dispatch_insert_dispatch.finalize();
  m_inner->insert_dispatch_insert_bindings.finalize();
}

void Db::create_cached_stmts() {
  m_inner->query_binary_by_hash = m_inner->db.prepare(
      "SELECT spirv FROM shader_binaries WHERE src_sha256 = ?1;");
  m_inner->query_dispatch_latency =
      m_inner->db.prepare("SELECT d.mean_latency_ns "
                          "FROM dispatches d "
                          "JOIN shader_binaries b ON b.id = d.binary_id "
                          "WHERE d.hash = ?1 "
                          "AND d.wg_x = ?2 "
                          "AND d.wg_y = ?3 "
                          "AND d.wg_z = ?4 "
                          "AND d.push_constant = ?5 "
                          "AND b.src_sha256 = ?6 "
                          "AND d.mean_latency_ns IS NOT NULL "
                          "LIMIT 1;");
  m_inner->query_binary_existence = m_inner->db.prepare(
      "SELECT id, spirv FROM shader_binaries WHERE src_sha256 = ?1;");
  m_inner->insert_binary = m_inner->db.prepare(
      "INSERT INTO shader_binaries(src_sha256, spirv) VALUES(?1, ?2);");
  m_inner->insert_dispatch_query_dispatch_existance =
      m_inner->db.prepare("SELECT id FROM dispatches "
                          "WHERE hash = ?1 "
                          "AND binary_id = ?2 "
                          "AND wg_x = ?3 "
                          "AND wg_y = ?4 "
                          "AND wg_z = ?5 "
                          "AND push_constant = ?6 "
                          "AND ( (input_bindings IS NULL AND ?7 IS NULL) "
                          "      OR input_bindings = ?7 ) "
                          "AND ( (output_bindings IS NULL AND ?8 IS NULL) "
                          "      OR output_bindings = ?8 ) "
                          "LIMIT 1;");
  m_inner->insert_dispatch_insert_dispatch = m_inner->db.prepare(
      "INSERT INTO dispatches("
      "binary_id, wg_x, wg_y, wg_z, push_constant, hash, "
      "operation, shader_name, config, memory_reads, memory_writes, flops, "
      "coopmat, fixed_subgroup_size, "
      "input_bindings, output_bindings"
      ") VALUES("
      "?1,?2,?3,?4,?5,?6,"
      "?7,?8,?9,?10,?11,?12,"
      "?13,?14,?15,?16"
      ");");
  m_inner->insert_dispatch_insert_bindings = m_inner->db.prepare(
      "INSERT INTO dispatch_bindings("
      "dispatch_id, idx, set_, binding, access, format, storage, "
      "byte_size, alignment, width, height, channels, dtype, is_param"
      ") VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14);");
}

memory::vector<uint32_t> Db::queryAllComputeDispatchIds() const {
  std::lock_guard lck{m_inner->mutex};
  auto stmt = m_inner->db.prepare("SELECT id FROM dispatches ORDER BY id ASC;");
  memory::vector<uint32_t> ids;
  while (stmt.next()) {
    ids.push_back(static_cast<uint32_t>(stmt.as_int64(0)));
  }
  return ids;
}

memory::vector<DbDispatchTimingInfo> Db::queryAllDispatchTimingInfos() const {
  std::lock_guard lck{m_inner->mutex};

  auto stmt =
      m_inner->db.prepare("SELECT d.id, "
                          "       COUNT(t.dispatch_id) AS sample_count, "
                          "       d.mean_latency_ns, "
                          "       d.std_derivation_ns "
                          "FROM dispatches d "
                          "LEFT JOIN timing_samples t "
                          "  ON t.dispatch_id = d.id "
                          "GROUP BY d.id "
                          "ORDER BY d.id ASC;");

  memory::vector<DbDispatchTimingInfo> out;

  while (stmt.next()) {
    DbDispatchTimingInfo info{};
    info.dispatch_id = static_cast<uint32_t>(stmt.as_int64(0));

    info.sample_count = static_cast<uint64_t>(stmt.as_int64(1));

    info.mean_latency_ns =
        stmt.is_null(2) ? 0 : static_cast<uint64_t>(stmt.as_int64(2));

    info.std_derivation_ns =
        stmt.is_null(3) ? 0 : static_cast<uint64_t>(stmt.as_int64(3));

    out.push_back(info);
  }

  return out;
}

bool Db::has_shader_binary(const SHA256 &srcHash) const {
  std::lock_guard lck{m_inner->mutex};

  auto &stmt = m_inner->query_binary_existence;
  stmt.reset();
  stmt.clear_bindings();

  stmt.bind_blob(1, srcHash.h, static_cast<int>(sizeof(uint32_t) * 8));

  return stmt.next();
}

memory::hash_map<SHA256, uint32_t> Db::query_in_memory_shader_cache() const {
  std::lock_guard lck{m_inner->mutex};
  memory::hash_map<SHA256, uint32_t> cache;
  cache.reserve(1 << 16);
  auto stmt =
      m_inner->db.prepare("SELECT id, src_sha256 FROM shader_binaries;");
  while (stmt.next()) {
    const uint32_t id = static_cast<uint32_t>(stmt.as_int64(0));
    auto blob = stmt.as_blob_view(1);
    if (blob.size() != sizeof(SHA256::h)) {
      throw std::runtime_error("Invalid SHA256 size in shader_binaries table");
    }
    SHA256 hash;
    std::memcpy(hash.h, blob.data(), sizeof(hash.h));
    cache.emplace(hash, id);
  }
  return cache;
}

memory::optional<uint32_t>
Db::query_shader_binary_id(const SHA256 &srcHash) const {
  std::lock_guard lck{m_inner->mutex};

  auto &stmt = m_inner->query_binary_existence;
  stmt.reset();
  stmt.clear_bindings();

  stmt.bind_blob(1, srcHash.h, static_cast<int>(sizeof(srcHash.h)));

  if (!stmt.next()) {
    return memory::nullopt;
  }

  return static_cast<uint32_t>(stmt.as_int64(0));
}

DbConvergenceInfo Db::query_convergence_info(uint64_t minSamples,
                                             double maxRelativeError) const {
  std::lock_guard lck{m_inner->mutex};
  const std::string sql = "SELECT "
                          "  COUNT(*) AS total_dispatch_count, "
                          "  COALESCE(SUM(sample_count), 0) AS total_samples, "
                          "  COALESCE(SUM(CASE "
                          "      WHEN sample_count >= ?1 "
                          "      THEN 1 ELSE 0 END), 0) "
                          "    AS converged_min_dispatches, "
                          "  COALESCE(SUM(CASE "
                          "      WHEN sample_count > 1 "
                          "       AND mean_latency_ns > 0 "
                          "       AND std_derivation_ns > 0 "
                          "       AND "
                          "         (std_derivation_ns * std_derivation_ns) "
                          "         <= (?2 * ?2) "
                          "            * (mean_latency_ns * mean_latency_ns) "
                          "            * sample_count "
                          "      THEN 1 ELSE 0 END), 0) "
                          "    AS converged_rel_dispatches "
                          "FROM ( "
                          "  SELECT "
                          "    d.id, "
                          "    d.mean_latency_ns, "
                          "    d.std_derivation_ns, "
                          "    COUNT(t.dispatch_id) AS sample_count "
                          "  FROM dispatches d "
                          "  LEFT JOIN timing_samples t "
                          "    ON t.dispatch_id = d.id "
                          "  GROUP BY d.id "
                          ");";
  auto stmt = m_inner->db.prepare(sql.c_str());
  stmt.bind_int64(1, static_cast<int64_t>(minSamples));
  stmt.bind_double(2, maxRelativeError);
  if (!stmt.next()) {
    throw std::runtime_error("Failed to compute convergence info");
  }
  DbConvergenceInfo out{};
  out.total_dispatch_count = static_cast<uint64_t>(stmt.as_int64(0));
  out.total_samples = static_cast<uint64_t>(stmt.as_int64(1));
  out.converged_min_dispatches = static_cast<uint64_t>(stmt.as_int64(2));
  out.converged_rel_dispatches = static_cast<uint64_t>(stmt.as_int64(3));
  return out;
}
} // namespace denox
