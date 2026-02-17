#include "denox/db/Db.hpp"
#include "denox/algorithm/hash_combine.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/common/version.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/db/DbIndex.hpp"
#include "denox/db/DbMapped.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
#include "flatbuffers/verifier.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <db.h>
#include <filesystem>
#include <fmt/format.h>
#include <mutex>
#include <ratio>
#include <stdexcept>

#include "sqlite3.h"

static constexpr bool USE_SQLITE = true;

namespace {

static inline void sqlite_check(int rc, sqlite3 *db, const char *what) {
  if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) {
    return;
  }
  const char *msg = db ? sqlite3_errmsg(db) : "no sqlite db";
  throw std::runtime_error(
      fmt::format("SQLite error in {}: rc={}, msg={}", what, rc, msg));
}

static inline void sqlite_exec(sqlite3 *db, const char *sql) {
  char *err = nullptr;
  int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
  if (rc != SQLITE_OK) {
    const char *msg = err ? err : sqlite3_errmsg(db);
    if (err)
      sqlite3_free(err);
    throw std::runtime_error(
        fmt::format("SQLite exec failed: rc={}, msg={}, sql={}", rc, msg, sql));
  }
}

struct Stmt {
  sqlite3_stmt *s = nullptr;
  Stmt() = default;
  Stmt(sqlite3 *db, const char *sql) {
    sqlite_check(sqlite3_prepare_v2(db, sql, -1, &s, nullptr), db, "prepare");
  }
  ~Stmt() {
    if (s)
      sqlite3_finalize(s);
  }
  Stmt(const Stmt &) = delete;
  Stmt &operator=(const Stmt &) = delete;
};

template <class S> static std::string_view as_sv(const S &s) {
  if constexpr (std::is_convertible_v<const S &, std::string_view>) {
    return std::string_view(s);
  } else {
    return std::string_view(s.data(), s.size());
  }
}

template <class Opt> static bool has_value_like(const Opt &o) {
  return o.has_value();
}

static bool col_is_null(sqlite3_stmt *s, int col) {
  return sqlite3_column_type(s, col) == SQLITE_NULL;
}

inline std::string col_text(sqlite3_stmt *s, int col) {
  const unsigned char *p = sqlite3_column_text(s, col);
  int n = sqlite3_column_bytes(s, col);
  if (!p || n <= 0)
    return {};
  return std::string(reinterpret_cast<const char *>(p),
                     reinterpret_cast<const char *>(p) + n);
}

inline std::vector<std::uint8_t> col_blob_u8(sqlite3_stmt *s, int col) {
  const void *p = sqlite3_column_blob(s, col);
  int n = sqlite3_column_bytes(s, col);
  if (!p || n <= 0)
    return {};
  const auto *b = reinterpret_cast<const std::uint8_t *>(p);
  return std::vector<std::uint8_t>(b, b + n);
}

inline void col_blob_exact(sqlite3_stmt *s, int col, void *dst,
                           int expected_bytes, const char *what, sqlite3 *db) {
  const void *p = sqlite3_column_blob(s, col);
  int n = sqlite3_column_bytes(s, col);
  if (!p || n != expected_bytes) {
    throw std::runtime_error(
        fmt::format("SQLite: invalid blob size for {}: expected {}, got {}",
                    what, expected_bytes, n));
  }
  std::memcpy(dst, p, static_cast<size_t>(expected_bytes));
}

inline std::vector<std::uint32_t> col_blob_u32(sqlite3_stmt *s, int col,
                                               const char *what, sqlite3 *db) {
  const void *p = sqlite3_column_blob(s, col);
  int n = sqlite3_column_bytes(s, col);
  if (!p || n <= 0)
    return {};
  if ((n % int(sizeof(std::uint32_t))) != 0) {
    throw std::runtime_error(fmt::format(
        "SQLite: blob for {} not multiple of 4 bytes (n={})", what, n));
  }
  size_t count = size_t(n) / sizeof(std::uint32_t);
  std::vector<std::uint32_t> out(count);
  std::memcpy(out.data(), p, size_t(n));
  return out;
}

inline std::uint64_t col_u64(sqlite3_stmt *s, int col) {
  // SQLite INTEGER is signed 64-bit.
  sqlite3_int64 v = sqlite3_column_int64(s, col);
  return static_cast<std::uint64_t>(v);
}

} // namespace

denox::Db denox::Db::open(const io::Path &path) {
  auto out = std::make_shared<DbMapped>();
  out->m_path = path;
  auto index = std::make_shared<DbIndex>();
  if (!path.exists()) {
    return Db{std::move(out), std::move(index)};
  }
  if (path.is_dir()) {
    DENOX_ERROR("invalid database path: {} is a directory", path);
    diag::invalid_argument();
  }
  out->m_path = path;

  const bool loadSqlite = USE_SQLITE;

  if (loadSqlite) {
    sqlite3 *db = nullptr;
    {
      int rc = sqlite3_open_v2(path.cstr(), &db, SQLITE_OPEN_READONLY, nullptr);
      if (rc != SQLITE_OK) {
        const char *msg = db ? sqlite3_errmsg(db) : "no sqlite db";
        if (db) {
          sqlite3_close(db);
        }

        throw std::runtime_error(fmt::format(
            "Failed to open database: {}: rc={}, msg={}", path, rc, msg));
      }
    }
    try {
      int user_version = 0;
      {
        Stmt st(db, "PRAGMA user_version;");
        int rc = sqlite3_step(st.s);
        if (rc != SQLITE_ROW) {
          // Common failure for "file is not a database"
          sqlite_check(rc, db, "step PRAGMA user_version");
        }
        user_version = sqlite3_column_int(st.s, 0);
      }
      if (user_version != 1) {
        throw std::runtime_error(fmt::format(
            "Unsupported sqlite db schema version {} (expected 1) in {}",
            user_version, path));
      }

      {
        Stmt st(db, "SELECT id, device, os, driver_version, denox_version, "
                    "denox_commit_hash, "
                    "start_timestamp, clock_mode, l2_warmup_iterations, "
                    "jit_warmup_iterations, measurement_iterations "
                    "FROM envs ORDER BY id;");

        std::vector<DbEnv> envs;
        while (true) {
          int rc = sqlite3_step(st.s);
          if (rc == SQLITE_DONE)
            break;
          sqlite_check(rc, db, "step envs");

          const int id = sqlite3_column_int(st.s, 0);
          if (id < 0) {
            throw std::runtime_error(
                fmt::format("SQLite: negative env id {}", id));
          }
          if (size_t(id) != envs.size()) {
            throw std::runtime_error(
                fmt::format("SQLite: env id {} not contigous (expected {})", id,
                            envs.size()));
          }

          DbEnv e{};
          e.device = col_text(st.s, 1);
          e.os = col_text(st.s, 2);
          e.driver_version = col_text(st.s, 3);
          e.denox_version = col_text(st.s, 4);
          e.denox_commit_hash = col_text(st.s, 5);
          e.start_timestamp =
              static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 6));
          e.clock_mode = static_cast<DbClockMode>(sqlite3_column_int(st.s, 7));
          e.l2_warmup_iterations =
              static_cast<std::uint16_t>(sqlite3_column_int(st.s, 8));
          e.jit_warmup_iterations =
              static_cast<std::uint16_t>(sqlite3_column_int(st.s, 9));
          e.measurement_iterations =
              static_cast<std::uint16_t>(sqlite3_column_int(st.s, 10));

          envs.push_back(std::move(e));
        }
        out->environments = std::move(envs);
      }

      {
        Stmt st(
            db,
            "SELECT id, src_sha256, spirv FROM shader_binaries ORDER BY id;");

        while (true) {
          int rc = sqlite3_step(st.s);
          if (rc == SQLITE_DONE)
            break;
          sqlite_check(rc, db, "step shader_binaries");

          const int id = sqlite3_column_int(st.s, 0);
          if (id < 0) {
            throw std::runtime_error(
                fmt::format("SQLite: negative shader_binaries id {}", id));
          }
          if (size_t(id) != out->binaries.size()) {
            throw std::runtime_error(fmt::format(
                "SQLite: shader_binaries id {} not contigous (expected {})", id,
                out->binaries.size()));
          }

          DbShaderBinary bin{};
          col_blob_exact(st.s, 1, bin.hash.h, int(sizeof(std::uint32_t) * 8),
                         "src_sha256", db);

          // spirv as blob of uint32 words
          {
            const void *p = sqlite3_column_blob(st.s, 2);
            int n = sqlite3_column_bytes(st.s, 2);
            if (!p || n <= 0 || (n % int(sizeof(std::uint32_t))) != 0) {
              throw std::runtime_error(fmt::format(
                  "SQLite: invalid spirv blob (bytes={}) for binary {}", n,
                  id));
            }
            const size_t words = size_t(n) / sizeof(std::uint32_t);
            bin.spvBinary.spv.resize(words);
            std::memcpy(bin.spvBinary.spv.data(), p, size_t(n));
          }

          // build index (same as your flatbuffer path)
          if (index->binary_index.contains(bin.hash)) {
            throw std::runtime_error(
                "SHA256 collision (binary_index). Database may be corrupted?");
          }
          index->binary_index[bin.hash] = out->binaries.size();

          out->binaries.push_back(std::move(bin));
        }
      }
      {
        Stmt st(db,
                "SELECT id, binary_id, wg_x, wg_y, wg_z, push_constant, hash, "
                "operation, shader_name, config, memory_reads, memory_writes, "
                "flops, "
                "coopmat, "
                "input_bindings, output_bindings, mean_latency_ns, "
                "std_derivation_ns "
                "FROM dispatches ORDER BY id;");

        while (true) {
          int rc = sqlite3_step(st.s);
          if (rc == SQLITE_DONE)
            break;
          sqlite_check(rc, db, "step dispatches");

          const int id = sqlite3_column_int(st.s, 0);
          if (id < 0) {
            throw std::runtime_error(
                fmt::format("SQLite: negative disaptch id {}", id));
          }
          if (size_t(id) != out->dispatches.size()) {
            throw std::runtime_error(fmt::format(
                "SQLite: dispatch id {} not contigous (expected {})", id,
                out->dispatches.size()));
          }

          DbComputeDispatch d{};
          d.binaryId = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 1));
          d.workgroupCountX =
              static_cast<std::uint32_t>(sqlite3_column_int(st.s, 2));
          d.workgroupCountY =
              static_cast<std::uint32_t>(sqlite3_column_int(st.s, 3));
          d.workgroupCountZ =
              static_cast<std::uint32_t>(sqlite3_column_int(st.s, 4));
          d.pushConstant = col_blob_u8(st.s, 5);
          d.hash = col_u64(st.s, 6);

          if (!col_is_null(st.s, 7))
            d.operation = col_text(st.s, 7);
          if (!col_is_null(st.s, 8))
            d.shader_name = col_text(st.s, 8);
          if (!col_is_null(st.s, 9))
            d.config = col_text(st.s, 9);

          if (!col_is_null(st.s, 10))
            d.memory_reads =
                static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 10));
          if (!col_is_null(st.s, 11))
            d.memory_writes =
                static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 11));
          if (!col_is_null(st.s, 12))
            d.flops =
                static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 12));
          if (!col_is_null(st.s, 13))
            d.coopmat = (sqlite3_column_int(st.s, 13) != 0);

          if (!col_is_null(st.s, 14)) {
            auto v = col_blob_u32(st.s, 14, "input_bindings", db);
            d.input_bindings.emplace(v.begin(), v.end());
          }
          if (!col_is_null(st.s, 15)) {
            auto v = col_blob_u32(st.s, 15, "output_bindings", db);
            d.output_bindings.emplace(v.begin(), v.end());
          }

          // timing summary: if present, create time struct; samples loaded
          // later
          if (!col_is_null(st.s, 16) || !col_is_null(st.s, 17)) {
            d.time.emplace();
            d.time->mean_latency_ns =
                !col_is_null(st.s, 16)
                    ? static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 16))
                    : 0;
            d.time->std_derivation_ns =
                !col_is_null(st.s, 17)
                    ? static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 17))
                    : 0;
            // samples filled later
          } else {
            d.time = std::nullopt;
          }

          // (bindings filled later)
          out->dispatches.push_back(std::move(d));

          // build bucket index
          index->dispatch_buckets[out->dispatches.back().hash].push_back(
              out->dispatches.size() - 1);
        }
      }

      // Validate binary_id range early (optional but useful)
      for (size_t i = 0; i < out->dispatches.size(); ++i) {
        if (out->dispatches[i].binaryId >= out->binaries.size()) {
          throw std::runtime_error(fmt::format(
              "SQLite: dispatch {} references invalid binary_id {} "
              "(binaries={})",
              i, out->dispatches[i].binaryId, out->binaries.size()));
        }
      }

      // ---- dispatch_bindings ----
      {
        Stmt st(db, "SELECT dispatch_id, idx, set_, binding, access, format, "
                    "storage, byte_size, alignment, "
                    "width, height, channels, dtype, is_param "
                    "FROM dispatch_bindings ORDER BY dispatch_id, idx;");

        while (true) {
          int rc = sqlite3_step(st.s);
          if (rc == SQLITE_DONE)
            break;
          sqlite_check(rc, db, "step dispatch_bindings");

          const int dispatch_id = sqlite3_column_int(st.s, 0);
          const int idx_in_disp = sqlite3_column_int(st.s, 1);
          if (dispatch_id < 0 ||
              size_t(dispatch_id) >= out->dispatches.size() ||
              idx_in_disp < 0) {
            throw std::runtime_error(
                fmt::format("SQLite: invalid dispatch_bindings row "
                            "(dispatch_id={}, idx={})",
                            dispatch_id, idx_in_disp));
          }

          auto &d = out->dispatches[size_t(dispatch_id)];
          const size_t bi = size_t(idx_in_disp);
          if (d.bindings.size() <= bi)
            d.bindings.resize(bi + 1);

          DbTensorBinding b{};
          b.set = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 2));
          b.binding = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 3));
          b.access = static_cast<Access>(sqlite3_column_int(st.s, 4));
          b.format = static_cast<TensorFormat>(sqlite3_column_int(st.s, 5));
          b.storage = static_cast<TensorStorage>(sqlite3_column_int(st.s, 6));
          b.byteSize =
              static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 7));
          b.alignment = static_cast<std::uint16_t>(sqlite3_column_int(st.s, 8));

          if (!col_is_null(st.s, 9))
            b.width = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 9));
          if (!col_is_null(st.s, 10))
            b.height = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 10));
          if (!col_is_null(st.s, 11))
            b.channels =
                static_cast<std::uint32_t>(sqlite3_column_int(st.s, 11));
          if (!col_is_null(st.s, 12))
            b.type = static_cast<TensorDataType>(sqlite3_column_int(st.s, 12));
          b.is_param = (sqlite3_column_int(st.s, 13) != 0);

          if (b.storage == TensorStorage::Optimal) {
            throw std::runtime_error(
                fmt::format("SQLite: invalid storage=Optimal, in "
                            "dispatch_bindings (dispatch_id={}, idx={})",
                            dispatch_id, idx_in_disp));
          }

          d.bindings[bi] = std::move(b);
        }
      }
      // ---- timing_samples ----
      {
        Stmt st(db, "SELECT dispatch_id, idx, timestamp, latency_ns, env "
                    "FROM timing_samples ORDER BY dispatch_id, idx;");

        while (true) {
          int rc = sqlite3_step(st.s);
          if (rc == SQLITE_DONE)
            break;
          sqlite_check(rc, db, "step timing_samples");

          const int dispatch_id = sqlite3_column_int(st.s, 0);
          const int idx = sqlite3_column_int(st.s, 1);
          if (dispatch_id < 0 ||
              size_t(dispatch_id) >= out->dispatches.size() || idx < 0) {
            throw std::runtime_error(fmt::format(
                "SQLite: invalid timing_samples row (dispatch_id={}, idx={})",
                dispatch_id, idx));
          }

          auto &d = out->dispatches[size_t(dispatch_id)];
          if (!d.time.has_value()) {
            // Should not happen if you only write samples when time exists, but
            // tolerate it.
            d.time.emplace();
            d.time->mean_latency_ns = 0;
            d.time->std_derivation_ns = 0;
          }

          if (d.time->samples.size() != size_t(idx)) {
            // keep strict: rows must be contiguous, ordered
            throw std::runtime_error(fmt::format(
                "SQLite: sample idx {} not contiguous (expected {}) "
                "for dispatch {}",
                idx, d.time->samples.size(), dispatch_id));
          }

          DbSample s{};
          s.timestamp =
              static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 2));
          s.latency_ns =
              static_cast<std::uint64_t>(sqlite3_column_int64(st.s, 3));
          s.env = static_cast<std::uint32_t>(sqlite3_column_int(st.s, 4));
          d.time->samples.push_back(std::move(s));
        }
      }
    } catch (const std::runtime_error &e) {
      if (db) {
        int rc = sqlite3_close(db);
        if (rc != SQLITE_OK) {
          DENOX_ERROR("SQLite close failed during error rewind: rc={}");
        }
        db = nullptr;
        throw; // <- rethrow
      }
    }
    if (db) {
      int rc = sqlite3_close(db);
      if (rc != SQLITE_OK) {
        throw std::runtime_error(fmt::format("SQLite failed to close: rc={}, "
                                             "msg={}",
                                             rc, sqlite3_errmsg(db)));
      }
      db = nullptr;
    }

    return Db(std::move(out), std::move(index));

  } else {
    auto file = io::File::open(path, io::File::OpenMode::Read);

    std::vector<std::uint8_t> buffer(file.size());
    file.read_exact(
        std::span{reinterpret_cast<std::byte *>(buffer.data()), buffer.size()});

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!denox::db::VerifyDbBuffer(verifier)) {
      DENOX_ERROR("invalid database format!");
      diag::invalid_argument();
    }

    const db::Db *db = db::GetDb(buffer.data());

    if (db->env() != nullptr) {
      const uint32_t env_count = db->env()->size();
      std::vector<DbEnv> envs;
      envs.reserve(env_count);
      for (uint32_t i = 0; i < env_count; ++i) {
        const auto *env = db->env()->Get(i);

        DbClockMode clock_mode;
        switch (env->clock_mode()) {
        case db::ClockMode_Unavailable:
          clock_mode = DbClockMode::Unavailable;
          break;
        case db::ClockMode_None:
          clock_mode = DbClockMode::None;
          break;
        case db::ClockMode_Base:
          clock_mode = DbClockMode::Base;
          break;
        case db::ClockMode_Maximum:
          clock_mode = DbClockMode::Maximum;
          break;
        }

        envs.push_back(DbEnv{
            .device = env->device()->str(),
            .os = env->device()->str(),
            .driver_version = env->driver_version()->str(),
            .denox_version = env->denox_version()->str(),
            .denox_commit_hash = env->denox_commit_hash()->str(),
            .start_timestamp = env->start_timestamp(),
            .clock_mode = clock_mode,
            .l2_warmup_iterations = env->l2_warmup_iterations(),
            .jit_warmup_iterations = env->jit_warmup_iterations(),
            .measurement_iterations = env->measurement_iterations(),
        });
      }
      out->environments = envs;
    }

    if (db->shader_binaries() != nullptr) {
      uint32_t binary_count = db->shader_binaries()->size();
      out->binaries.resize(binary_count);
      for (uint32_t i = 0; i < binary_count; ++i) {
        const db::ShaderBinary *binary = db->shader_binaries()->Get(i);
        out->binaries[i].spvBinary.spv.assign(binary->spirv()->begin(),
                                              binary->spirv()->end());
        assert(binary->src_sha256()->size() == 8);
        std::memcpy(out->binaries[i].hash.h, binary->src_sha256()->data(),

                    sizeof(uint32_t) * 8);
        // build index.
        if (index->binary_index.contains(out->binaries[i].hash)) {
          DENOX_ERROR(
              "glsl source SHA256 collision. This should be incredibly "
              "unlikely please contact us, by creating a github issue.");
          diag::invalid_state();
        }
        index->binary_index[out->binaries[i].hash] = i;
      }
    }

    if (db->dispatches() != nullptr) {
      uint32_t dispatchCount = db->dispatches()->size();
      out->dispatches.resize(dispatchCount);
      for (uint32_t i = 0; i < dispatchCount; ++i) {
        const db::ComputeDispatch *dispatch = db->dispatches()->Get(i);

        auto &out_dispatch = out->dispatches[i];

        out_dispatch.binaryId = dispatch->binary_id();
        out_dispatch.workgroupCountX = dispatch->workgroup_count_x();
        out_dispatch.workgroupCountY = dispatch->workgroup_count_y();
        out_dispatch.workgroupCountZ = dispatch->workgroup_count_z();
        out_dispatch.pushConstant.assign(dispatch->push_constant()->begin(),
                                         dispatch->push_constant()->end());
        out_dispatch.hash = dispatch->hash();

        uint32_t bindingCount = dispatch->bindings()->size();
        out->dispatches[i].bindings.resize(bindingCount);
        for (uint32_t b = 0; b < bindingCount; ++b) {
          const db::TensorBinding *binding = dispatch->bindings()->Get(b);
          out_dispatch.bindings[b].binding = binding->binding();
          out_dispatch.bindings[b].set = binding->set();
          switch (binding->access()) {
          case db::Access_ReadOnly:
            out_dispatch.bindings[b].access = Access::ReadOnly;
            break;
          case db::Access_WriteOnly:
            out_dispatch.bindings[b].access = Access::WriteOnly;
            break;
          case db::Access_ReadWrite:
            out_dispatch.bindings[b].access = Access::ReadWrite;
            break;
          }
          out_dispatch.bindings[b].byteSize = binding->tensor_byte_size();
          out_dispatch.bindings[b].alignment =
              static_cast<uint16_t>(binding->tensor_min_align());
          switch (binding->format()) {
          case db::TensorFormat_UNKNOWN:
            out_dispatch.bindings[b].format = TensorFormat::Optimal;
            break;
          case db::TensorFormat_SSBO_HWC:
            out_dispatch.bindings[b].format = TensorFormat::SSBO_HWC;
            break;
          case db::TensorFormat_SSBO_CHW:
            out_dispatch.bindings[b].format = TensorFormat::SSBO_CHW;
            break;
          case db::TensorFormat_SSBO_CHWC8:
            out_dispatch.bindings[b].format = TensorFormat::SSBO_CHWC8;
            break;
          case db::TensorFormat_TEX_RGBA:
            out_dispatch.bindings[b].format = TensorFormat::TEX_RGBA;
            break;
          case db::TensorFormat_TEX_RGB:
            out_dispatch.bindings[b].format = TensorFormat::TEX_RGB;
            break;
          case db::TensorFormat_TEX_RG:
            out_dispatch.bindings[b].format = TensorFormat::TEX_RG;
            break;
          case db::TensorFormat_TEX_R:
            out_dispatch.bindings[b].format = TensorFormat::TEX_R;
            break;
          }

          switch (binding->storage()) {
          case db::TensorStorage_StorageBuffer:
            out_dispatch.bindings[b].storage = TensorStorage::StorageBuffer;
            break;
          case db::TensorStorage_StorageImage:
            out_dispatch.bindings[b].storage = TensorStorage::StorageImage;
            break;
          case db::TensorStorage_SampledStorageImage:
            out_dispatch.bindings[b].storage =
                TensorStorage::SampledStorageImage;
            break;
          }

          if (binding->info() != nullptr) {
            const auto *info = binding->info();
            if (info->width()) {
              out_dispatch.bindings[b].width = info->width();
            }
            if (info->height()) {
              out_dispatch.bindings[b].height = info->height();
            }
            if (info->channels()) {
              out_dispatch.bindings[b].channels = info->channels();
            }
            switch (info->dtype()) {
            case db::TensorDataType_Float16:
              out_dispatch.bindings[b].type = TensorDataType::Float16;
              break;
            case db::TensorDataType_Unknown:
              break;
            }
            out_dispatch.bindings[b].is_param = info->is_param();
          }
        }

        if (dispatch->time() != nullptr) {
          out_dispatch.time.emplace();
          std::vector<DbSample> samples;
          const uint32_t sample_count = dispatch->time()->samples()->size();
          samples.reserve(sample_count);
          for (uint32_t i = 0; i < sample_count; ++i) {
            const auto &sample = dispatch->time()->samples()->Get(i);
            samples.push_back(DbSample{
                .timestamp = sample->timestamp(),
                .latency_ns = sample->latency_ns(),
                .env = sample->env(),
            });
          }
          out_dispatch.time->samples = samples;
          out_dispatch.time->mean_latency_ns =
              dispatch->time()->mean_latency_ns();
          out_dispatch.time->std_derivation_ns =
              dispatch->time()->std_derivation_ns();
        }

        if (dispatch->info() != nullptr) {
          const auto *info = dispatch->info();
          if (info->operation()) {
            out_dispatch.operation = info->operation()->str();
          }
          if (info->shader_name()) {
            out_dispatch.shader_name = info->shader_name()->str();
          }
          if (info->config()) {
            out_dispatch.config = info->config()->str();
          }
          if (info->memory_reads()) {
            out_dispatch.memory_reads = info->memory_reads();
          }
          if (info->memory_writes()) {
            out_dispatch.memory_writes = info->memory_writes();
          }
          if (info->flops()) {
            out_dispatch.flops = info->flops();
          }
          out_dispatch.coopmat = info->coopmat();
          if (info->input_bindings()) {
            out_dispatch.input_bindings.emplace(info->input_bindings()->begin(),
                                                info->input_bindings()->end());
          }
          if (info->output_bindings()) {
            out_dispatch.output_bindings.emplace(
                info->output_bindings()->begin(),
                info->output_bindings()->end());
          }
        }

        // build index.
        auto &bucket = index->dispatch_buckets[out_dispatch.hash];
        bucket.push_back(i);
      }
    }

    return Db(std::move(out), std::move(index));
  }
}

bool denox::Db::atomic_writeback() const {
  if (m_db->m_path.empty()) {
    return false;
  }

  // fmt::println("env-count: {}", m_db->environments.size());
  // fmt::println("binaries: {}", m_db->binaries.size());
  // fmt::println("dispatches: {}", m_db->dispatches.size());

  static std::mutex write_back_lock;
  std::lock_guard lck{write_back_lock};

  const bool writeSqlite = USE_SQLITE;

  if (!writeSqlite) {
    io::Path tmpPath = m_db->m_path.with_extension("db.tmp");

    flatbuffers::FlatBufferBuilder fbb(1 << 16);
    const size_t env_count = m_db->environments.size();
    std::vector<flatbuffers::Offset<db::BenchEnv>> envs;
    envs.reserve(env_count);
    for (size_t i = 0; i < env_count; ++i) {
      const auto &env = m_db->environments[i];

      db::ClockMode clockMode = db::ClockMode_Unavailable;
      switch (env.clock_mode) {
      case DbClockMode::Unavailable:
        break;
      case DbClockMode::None:
        clockMode = db::ClockMode_None;
        break;
      case DbClockMode::Base:
        clockMode = db::ClockMode_Base;
        break;
      case DbClockMode::Maximum:
        clockMode = db::ClockMode_Maximum;
        break;
      }

      envs.push_back(db::CreateBenchEnv(
          fbb, fbb.CreateString(env.device), fbb.CreateString(env.os),
          fbb.CreateString(env.driver_version),
          fbb.CreateString(env.denox_version),
          fbb.CreateString(env.denox_commit_hash), env.start_timestamp,
          clockMode, env.l2_warmup_iterations, env.jit_warmup_iterations,
          env.measurement_iterations));
    }
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<db::BenchEnv>>>
        envsVec = fbb.CreateVector(envs);

    std::vector<flatbuffers::Offset<db::ShaderBinary>> binaries;
    binaries.reserve(m_db->binaries.size());
    for (const DbShaderBinary &binary : m_db->binaries) {
      binaries.push_back(db::CreateShaderBinary(
          fbb, fbb.CreateVector<uint32_t>(binary.hash.h, 8),
          fbb.CreateVector<uint32_t>(binary.spvBinary.spv)));
    }
    auto binariesVec = fbb.CreateVector(binaries);

    std::vector<flatbuffers::Offset<db::ComputeDispatch>> dispatches;
    dispatches.reserve(m_db->dispatches.size());

    for (const auto &dispatch : m_db->dispatches) {
      std::vector<flatbuffers::Offset<db::TensorBinding>> bindings;
      bindings.reserve(dispatch.bindings.size());
      for (const auto &binding : dispatch.bindings) {
        db::Access access;
        switch (binding.access) {
        case Access::ReadOnly:
          access = db::Access_ReadOnly;
          break;
        case Access::WriteOnly:
          access = db::Access_WriteOnly;
          break;
        case Access::ReadWrite:
          access = db::Access_ReadWrite;
          break;
        default:
          diag::unreachable();
        }
        db::TensorFormat format;
        switch (binding.format) {
        case TensorFormat::Optimal:
          format = db::TensorFormat_UNKNOWN;
          break;
        case TensorFormat::SSBO_HWC:
          format = db::TensorFormat_SSBO_HWC;
          break;
        case TensorFormat::SSBO_CHW:
          format = db::TensorFormat_SSBO_CHW;
          break;
        case TensorFormat::SSBO_CHWC8:
          format = db::TensorFormat_SSBO_CHWC8;
          break;
        case TensorFormat::TEX_RGBA:
          format = db::TensorFormat_TEX_RGBA;
          break;
        case TensorFormat::TEX_RGB:
          format = db::TensorFormat_TEX_RGB;
          break;
        case TensorFormat::TEX_RG:
          format = db::TensorFormat_TEX_RG;
          break;
        case TensorFormat::TEX_R:
          format = db::TensorFormat_TEX_R;
          break;
        }
        db::TensorStorage storage;
        switch (binding.storage) {
        case TensorStorage::Optimal:
          assert(false && "trying to serialize TensorStorage::Optimal");
          diag::invalid_state();
        case TensorStorage::StorageBuffer:
          storage = db::TensorStorage_StorageBuffer;
          break;
        case TensorStorage::StorageImage:
          storage = db::TensorStorage_StorageImage;
          break;
        case TensorStorage::SampledStorageImage:
          storage = db::TensorStorage_SampledStorageImage;
          break;
        }

        flatbuffers::Offset<db::TensorBindingInfo> info;
        if (binding.width.has_value() || binding.height.has_value() ||
            binding.channels.has_value() || binding.type.has_value() ||
            binding.is_param) {
          db::TensorDataType dtype = db::TensorDataType_Unknown;
          if (binding.type.has_value()) {
            switch (*binding.type) {
            case TensorDataType::Auto:
              break;
            case TensorDataType::Float16:
              dtype = db::TensorDataType_Float16;
              break;
            case TensorDataType::Float32:
              diag::not_implemented();
            case TensorDataType::Float64:
              diag::not_implemented();
            }
          }

          info = db::CreateTensorBindingInfo(
              fbb, binding.width.value_or(0), binding.height.value_or(0),
              binding.channels.value_or(0), dtype, binding.is_param);
        }

        bindings.push_back(db::CreateTensorBinding(
            fbb, binding.set, binding.binding, access, format, storage,
            binding.byteSize, binding.alignment, info));
      }
      flatbuffers::Offset<db::Timing> time = 0;
      if (dispatch.time.has_value()) {
        size_t sample_count = dispatch.time->samples.size();
        std::vector<flatbuffers::Offset<db::Sample>> samples;
        samples.reserve(sample_count);
        for (uint32_t i = 0; i < sample_count; ++i) {
          samples.push_back(
              db::CreateSample(fbb, dispatch.time->samples[i].timestamp,
                               dispatch.time->samples[i].latency_ns,
                               dispatch.time->samples[i].env));
        }

        time = db::CreateTiming(fbb, fbb.CreateVector(samples),
                                dispatch.time->mean_latency_ns,
                                dispatch.time->std_derivation_ns);
      }

      flatbuffers::Offset<db::ComputeDispatchInfo> info;
      if (dispatch.operation.has_value() || dispatch.shader_name.has_value() ||
          dispatch.config.has_value() || dispatch.memory_reads.has_value() ||
          dispatch.memory_writes.has_value() || dispatch.flops.has_value() ||
          dispatch.coopmat.has_value() || dispatch.input_bindings.has_value() ||
          dispatch.output_bindings.has_value()) {

        flatbuffers::Offset<flatbuffers::String> operation;
        flatbuffers::Offset<flatbuffers::String> shader_name;
        flatbuffers::Offset<flatbuffers::String> config;
        uint64_t memory_reads = dispatch.memory_reads.value_or(0);
        uint64_t memory_writes = dispatch.memory_writes.value_or(0);
        uint64_t flops = dispatch.flops.value_or(0);
        bool coopmat = dispatch.coopmat.value_or(false);
        flatbuffers::Offset<flatbuffers::Vector<uint32_t>> input_bindings;
        flatbuffers::Offset<flatbuffers::Vector<uint32_t>> output_bindings;

        if (dispatch.operation) {
          operation = fbb.CreateString(*dispatch.operation);
        }
        if (dispatch.shader_name) {
          shader_name = fbb.CreateString(*dispatch.shader_name);
        }
        if (dispatch.config) {
          config = fbb.CreateString(*dispatch.config);
        }
        if (dispatch.input_bindings) {
          input_bindings = fbb.CreateVector(*dispatch.input_bindings);
        }
        if (dispatch.output_bindings) {
          output_bindings = fbb.CreateVector(*dispatch.output_bindings);
        }

        info = db::CreateComputeDispatchInfo(
            fbb, operation, shader_name, config, memory_reads, memory_writes,
            flops, coopmat, input_bindings, output_bindings);
      }

      dispatches.push_back(db::CreateComputeDispatch(
          fbb, dispatch.binaryId, dispatch.workgroupCountX,
          dispatch.workgroupCountY, dispatch.workgroupCountZ,
          fbb.CreateVector<uint8_t>(dispatch.pushConstant), dispatch.hash,
          fbb.CreateVector(bindings), time, info));
    }

    auto dispatchesVec = fbb.CreateVector(dispatches);

    db::DbBuilder builder(fbb);
    builder.add_version(0);
    builder.add_shader_binaries(binariesVec);
    builder.add_dispatches(dispatchesVec);
    builder.add_env(envsVec);
    auto db = builder.Finish();
    denox::db::FinishDbBuffer(fbb, db);

    flatbuffers::DetachedBuffer detachedBuffer = fbb.Release();

    flatbuffers::Verifier verifier(detachedBuffer.data(),
                                   detachedBuffer.size());
    if (!db::VerifyDbBuffer(verifier)) {
      DENOX_ERROR("Failed to write db to \'{}\'", m_db->m_path);
      diag::invalid_state();
    }

    io::File tmpFile = io::File::open(
        tmpPath, io::File::OpenMode::Create | io::File::OpenMode::Truncate |
                     io::File::OpenMode::Write);
    tmpFile.write_exact(
        std::span{reinterpret_cast<const std::byte *>(detachedBuffer.data()),
                  detachedBuffer.size()});

    std::filesystem::rename(tmpPath.str(), m_db->m_path.str());
  } else { // sqllite writeback
    io::Path tmpPath = m_db->m_path.with_extension("sqlite.tmp");
    {
      std::error_code ec;
      std::filesystem::remove(tmpPath.str(), ec); // remove old tmp file.
    }

    sqlite3 *db = nullptr;
    {
      int rc =
          sqlite3_open_v2(tmpPath.cstr(), &db,
                          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
      if (rc != SQLITE_OK) {
        const char *msg = db ? sqlite3_errmsg(db) : "no sqlite db";
        if (db)
          sqlite3_close(db);
        throw std::runtime_error(
            fmt::format("Failed to open database: rc={}, msg= {}", rc, msg));
      }
    }
    try {

      // settings
      sqlite_exec(db, "PRAGMA journal_mode=OFF;");
      sqlite_exec(db, "PRAGMA synchronous=OFF;");
      sqlite_exec(db, "PRAGMA temp_store=MEMORY;");
      sqlite_exec(db, "PRAGMA locking_mode=EXCLUSIVE;");

      // versioning.
      sqlite_exec(db, "PRAGMA user_version=1;");

      sqlite_exec(db, "CREATE TABLE envs ("
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

      sqlite_exec(db, "CREATE TABLE shader_binaries ("
                      "  id INTEGER PRIMARY KEY,"
                      "  src_sha256 BLOB NOT NULL,"
                      "  spirv BLOB NOT NULL"
                      ");");
      sqlite_exec(db, "CREATE UNIQUE INDEX shader_binaries_sha_idx ON "
                      "shader_binaries(src_sha256);");

      sqlite_exec(db, "CREATE TABLE dispatches ("
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
                      "  input_bindings BLOB,"
                      "  output_bindings BLOB,"
                      "  mean_latency_ns INTEGER,"
                      "  std_derivation_ns INTEGER"
                      ");");
      sqlite_exec(db, "CREATE INDEX dispatches_hash_idx ON dispatches(hash);");

      sqlite_exec(db, "CREATE TABLE dispatch_bindings ("
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
                      "  PRIMARY KEY(dispatch_id, idx)"
                      ");");

      sqlite_exec(db, "CREATE TABLE timing_samples ("
                      "  dispatch_id INTEGER NOT NULL,"
                      "  idx INTEGER NOT NULL,"
                      "  timestamp INTEGER NOT NULL,"
                      "  latency_ns INTEGER NOT NULL,"
                      "  env INTEGER NOT NULL,"
                      "  PRIMARY KEY(dispatch_id, idx)"
                      ");");

      sqlite_exec(db, "BEGIN IMMEDIATE;");

      {
        Stmt ins_env(
            db,
            "INSERT INTO envs("
            "id, device, os, driver_version, denox_version, denox_commit_hash, "
            "start_timestamp, clock_mode,"
            "l2_warmup_iterations, jit_warmup_iterations, "
            "measurement_iterations"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?);");

        Stmt ins_bin(db, "INSERT INTO shader_binaries(id, src_sha256, spirv) "
                         "VALUES (?,?,?);");

        Stmt ins_dispatch(
            db, "INSERT INTO dispatches("
                "id, binary_id, wg_x, wg_y, wg_z, push_constant, hash,"
                "operation, shader_name, config, memory_reads, memory_writes, "
                "flops, "
                "coopmat,"
                "input_bindings, output_bindings, mean_latency_ns, "
                "std_derivation_ns"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);");

        Stmt ins_binding(db, "INSERT INTO dispatch_bindings("
                             "dispatch_id, idx, set_, binding, access, format, "
                             "storage, byte_size, alignment,"
                             "width, height, channels, dtype, is_param"
                             ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?);");

        Stmt ins_sample(db, "INSERT INTO timing_samples(dispatch_id, idx, "
                            "timestamp, latency_ns, env) VALUES (?,?,?,?,?);");

        for (size_t i = 0; i < m_db->environments.size(); ++i) {
          const auto &e = m_db->environments[i];
          sqlite3_reset(ins_env.s);
          sqlite3_clear_bindings(ins_env.s);

          sqlite_check(
              sqlite3_bind_int64(ins_env.s, 1, static_cast<sqlite3_int64>(i)),
              db, "bind env id");
          sqlite_check(sqlite3_bind_text(ins_env.s, 2, e.device.c_str(), -1,
                                         SQLITE_TRANSIENT),
                       db, "bind env device");
          sqlite_check(sqlite3_bind_text(ins_env.s, 3, e.os.c_str(), -1,
                                         SQLITE_TRANSIENT),
                       db, "bind env os");
          sqlite_check(sqlite3_bind_text(ins_env.s, 4, e.driver_version.c_str(),
                                         -1, SQLITE_TRANSIENT),
                       db, "bind env driver_version");
          sqlite_check(sqlite3_bind_text(ins_env.s, 5, e.denox_version.c_str(),
                                         -1, SQLITE_TRANSIENT),
                       db, "bind env denox_version");
          sqlite_check(sqlite3_bind_text(ins_env.s, 6,
                                         e.denox_commit_hash.c_str(), -1,
                                         SQLITE_TRANSIENT),
                       db, "bind env denox_commit_hash");
          sqlite_check(
              sqlite3_bind_int64(ins_env.s, 7,
                                 static_cast<sqlite3_int64>(e.start_timestamp)),
              db, "bind env start_timestamp");
          sqlite_check(
              sqlite3_bind_int(ins_env.s, 8, static_cast<int>(e.clock_mode)),
              db, "bind env clock_mode");
          sqlite_check(
              sqlite3_bind_int(ins_env.s, 9,
                               static_cast<int>(e.l2_warmup_iterations)),
              db, "bind env l2");
          sqlite_check(
              sqlite3_bind_int(ins_env.s, 10,
                               static_cast<int>(e.jit_warmup_iterations)),
              db, "bind env jit");
          sqlite_check(
              sqlite3_bind_int(ins_env.s, 11,
                               static_cast<int>(e.measurement_iterations)),
              db, "bind env meas");

          sqlite_check(sqlite3_step(ins_env.s), db, "step insert env");
        }
        // --- shader binaries ---
        for (size_t i = 0; i < m_db->binaries.size(); ++i) {
          const auto &b = m_db->binaries[i];

          sqlite3_reset(ins_bin.s);
          sqlite3_clear_bindings(ins_bin.s);

          sqlite_check(
              sqlite3_bind_int64(ins_bin.s, 1, static_cast<sqlite3_int64>(i)),
              db, "bind bin id");

          // SHA256: 8*u32 = 32 bytes
          sqlite_check(sqlite3_bind_blob(ins_bin.s, 2, b.hash.h,
                                         static_cast<int>(sizeof(uint32_t) * 8),
                                         SQLITE_TRANSIENT),
                       db, "bind bin sha");

          // SPIR-V: vector<uint32_t> -> raw bytes
          const void *spv_ptr =
              b.spvBinary.spv.empty() ? nullptr : b.spvBinary.spv.data();
          const int spv_len =
              static_cast<int>(b.spvBinary.spv.size() * sizeof(uint32_t));
          sqlite_check(sqlite3_bind_blob(ins_bin.s, 3, spv_ptr, spv_len,
                                         SQLITE_TRANSIENT),
                       db, "bind bin spirv");

          sqlite_check(sqlite3_step(ins_bin.s), db, "step insert bin");
        }

        // --- dispatches + bindings + samples ---
        for (size_t di = 0; di < m_db->dispatches.size(); ++di) {
          const auto &d = m_db->dispatches[di];

          sqlite3_reset(ins_dispatch.s);
          sqlite3_clear_bindings(ins_dispatch.s);

          sqlite_check(sqlite3_bind_int64(ins_dispatch.s, 1,
                                          static_cast<sqlite3_int64>(di)),
                       db, "bind dispatch id");
          sqlite_check(
              sqlite3_bind_int(ins_dispatch.s, 2, static_cast<int>(d.binaryId)),
              db, "bind dispatch binary_id");
          sqlite_check(sqlite3_bind_int(ins_dispatch.s, 3,
                                        static_cast<int>(d.workgroupCountX)),
                       db, "bind wg_x");
          sqlite_check(sqlite3_bind_int(ins_dispatch.s, 4,
                                        static_cast<int>(d.workgroupCountY)),
                       db, "bind wg_y");
          sqlite_check(sqlite3_bind_int(ins_dispatch.s, 5,
                                        static_cast<int>(d.workgroupCountZ)),
                       db, "bind wg_z");

          const void *pc_ptr =
              d.pushConstant.empty() ? nullptr : d.pushConstant.data();
          const int pc_len = static_cast<int>(d.pushConstant.size());
          sqlite_check(sqlite3_bind_blob(ins_dispatch.s, 6, pc_ptr, pc_len,
                                         SQLITE_TRANSIENT),
                       db, "bind push_constant");

          sqlite_check(sqlite3_bind_int64(ins_dispatch.s, 7,
                                          static_cast<sqlite3_int64>(d.hash)),
                       db, "bind hash");

          // Optional text/info
          if (d.operation.has_value()) {
            auto sv = as_sv(*d.operation);
            sqlite_check(sqlite3_bind_text(ins_dispatch.s, 8, sv.data(),
                                           static_cast<int>(sv.size()),
                                           SQLITE_TRANSIENT),
                         db, "bind operation");
          } else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 8), db,
                         "bind operation null");

          if (d.shader_name.has_value()) {
            auto sv = as_sv(*d.shader_name);
            sqlite_check(sqlite3_bind_text(ins_dispatch.s, 9, sv.data(),
                                           static_cast<int>(sv.size()),
                                           SQLITE_TRANSIENT),
                         db, "bind shader_name");
          } else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 9), db,
                         "bind shader_name null");

          if (d.config.has_value()) {
            auto sv = as_sv(*d.config);
            sqlite_check(sqlite3_bind_text(ins_dispatch.s, 10, sv.data(),
                                           static_cast<int>(sv.size()),
                                           SQLITE_TRANSIENT),
                         db, "bind config");
          } else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 10), db,
                         "bind config null");

          if (d.memory_reads.has_value())
            sqlite_check(
                sqlite3_bind_int64(ins_dispatch.s, 11,
                                   static_cast<sqlite3_int64>(*d.memory_reads)),
                db, "bind memory_reads");
          else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 11), db,
                         "bind memory_reads null");

          if (d.memory_writes.has_value())
            sqlite_check(sqlite3_bind_int64(
                             ins_dispatch.s, 12,
                             static_cast<sqlite3_int64>(*d.memory_writes)),
                         db, "bind memory_writes");
          else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 12), db,
                         "bind memory_writes null");

          if (d.flops.has_value())
            sqlite_check(
                sqlite3_bind_int64(ins_dispatch.s, 13,
                                   static_cast<sqlite3_int64>(*d.flops)),
                db, "bind flops");
          else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 13), db,
                         "bind flops null");

          if (d.coopmat.has_value())
            sqlite_check(
                sqlite3_bind_int(ins_dispatch.s, 14, *d.coopmat ? 1 : 0), db,
                "bind coopmat");
          else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 14), db,
                         "bind coopmat null");

          // input/output bindings as packed uint32 blob
          if (d.input_bindings.has_value()) {
            const auto &v = *d.input_bindings;
            const void *p = v.empty() ? nullptr : v.data();
            const int n = static_cast<int>(v.size() * sizeof(uint32_t));
            sqlite_check(
                sqlite3_bind_blob(ins_dispatch.s, 15, p, n, SQLITE_TRANSIENT),
                db, "bind input_bindings");
          } else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 15), db,
                         "bind input_bindings null");

          if (d.output_bindings.has_value()) {
            const auto &v = *d.output_bindings;
            const void *p = v.empty() ? nullptr : v.data();
            const int n = static_cast<int>(v.size() * sizeof(uint32_t));
            sqlite_check(
                sqlite3_bind_blob(ins_dispatch.s, 16, p, n, SQLITE_TRANSIENT),
                db, "bind output_bindings");
          } else
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 16), db,
                         "bind output_bindings null");

          // timing summary
          if (d.time.has_value()) {
            sqlite_check(sqlite3_bind_int64(ins_dispatch.s, 17,
                                            static_cast<sqlite3_int64>(
                                                d.time->mean_latency_ns)),
                         db, "bind mean_latency_ns");
            sqlite_check(sqlite3_bind_int64(ins_dispatch.s, 18,
                                            static_cast<sqlite3_int64>(
                                                d.time->std_derivation_ns)),
                         db, "bind std_derivation_ns");
          } else {
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 17), db,
                         "bind mean_latency_ns null");
            sqlite_check(sqlite3_bind_null(ins_dispatch.s, 18), db,
                         "bind std_derivation_ns null");
          }

          sqlite_check(sqlite3_step(ins_dispatch.s), db,
                       "step insert dispatch");

          // bindings rows
          for (size_t bi = 0; bi < d.bindings.size(); ++bi) {
            const auto &b = d.bindings[bi];
            if (b.storage == TensorStorage::Optimal) {
              DENOX_ERROR("Trying to serialize TensorStorage::Optimal");
              diag::invalid_state();
            }

            sqlite3_reset(ins_binding.s);
            sqlite3_clear_bindings(ins_binding.s);

            sqlite_check(sqlite3_bind_int64(ins_binding.s, 1,
                                            static_cast<sqlite3_int64>(di)),
                         db, "bind binding dispatch_id");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 2, static_cast<int>(bi)), db,
                "bind binding idx");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 3, static_cast<int>(b.set)), db,
                "bind binding set_");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 4, static_cast<int>(b.binding)),
                db, "bind binding binding");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 5, static_cast<int>(b.access)),
                db, "bind binding access");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 6, static_cast<int>(b.format)),
                db, "bind binding format");
            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 7, static_cast<int>(b.storage)),
                db, "bind binding storage");
            sqlite_check(
                sqlite3_bind_int64(ins_binding.s, 8,
                                   static_cast<sqlite3_int64>(b.byteSize)),
                db, "bind binding byte_size");
            sqlite_check(sqlite3_bind_int(ins_binding.s, 9,
                                          static_cast<int>(b.alignment)),
                         db, "bind binding alignment");

            if (b.width.has_value())
              sqlite_check(sqlite3_bind_int(ins_binding.s, 10,
                                            static_cast<int>(*b.width)),
                           db, "bind width");
            else
              sqlite_check(sqlite3_bind_null(ins_binding.s, 10), db,
                           "bind width null");
            if (b.height.has_value())
              sqlite_check(sqlite3_bind_int(ins_binding.s, 11,
                                            static_cast<int>(*b.height)),
                           db, "bind height");
            else
              sqlite_check(sqlite3_bind_null(ins_binding.s, 11), db,
                           "bind height null");
            if (b.channels.has_value())
              sqlite_check(sqlite3_bind_int(ins_binding.s, 12,
                                            static_cast<int>(*b.channels)),
                           db, "bind channels");
            else
              sqlite_check(sqlite3_bind_null(ins_binding.s, 12), db,
                           "bind channels null");
            if (b.type.has_value())
              sqlite_check(sqlite3_bind_int(ins_binding.s, 13,
                                            static_cast<int>(*b.type)),
                           db, "bind dtype");
            else
              sqlite_check(sqlite3_bind_null(ins_binding.s, 13), db,
                           "bind dtype null");

            sqlite_check(
                sqlite3_bind_int(ins_binding.s, 14, b.is_param ? 1 : 0), db,
                "bind is_param");

            sqlite_check(sqlite3_step(ins_binding.s), db,
                         "step insert binding");
          }

          // timing samples rows (if any)
          if (d.time.has_value()) {
            const auto &t = *d.time;
            for (size_t si = 0; si < t.samples.size(); ++si) {
              const auto &s = t.samples[si];

              sqlite3_reset(ins_sample.s);
              sqlite3_clear_bindings(ins_sample.s);

              sqlite_check(sqlite3_bind_int64(ins_sample.s, 1,
                                              static_cast<sqlite3_int64>(di)),
                           db, "bind sample dispatch_id");
              sqlite_check(
                  sqlite3_bind_int(ins_sample.s, 2, static_cast<int>(si)), db,
                  "bind sample idx");
              sqlite_check(
                  sqlite3_bind_int64(ins_sample.s, 3,
                                     static_cast<sqlite3_int64>(s.timestamp)),
                  db, "bind sample timestamp");
              sqlite_check(
                  sqlite3_bind_int64(ins_sample.s, 4,
                                     static_cast<sqlite3_int64>(s.latency_ns)),
                  db, "bind sample latency_ns");
              sqlite_check(
                  sqlite3_bind_int(ins_sample.s, 5, static_cast<int>(s.env)),
                  db, "bind sample env");

              sqlite_check(sqlite3_step(ins_sample.s), db,
                           "step insert sample");
            }
          }
        }

        sqlite_exec(db, "COMMIT;");
      }
    } catch (const std::exception &e) {
      if (db) {
        int rc = sqlite3_close(db);
        if (rc != SQLITE_OK) {
          DENOX_ERROR("SQLite close failed during error rewind: rc={}");
        }
        db = nullptr;
        throw;
      }
    }
    if (db) {
      int rc = sqlite3_close(db);
      if (rc != SQLITE_OK) {
        throw std::runtime_error(fmt::format("SQLite failed to close: rc={}, "
                                             "msg={}",
                                             rc, sqlite3_errmsg(db)));
      }
      db = nullptr;
    }

    std::error_code ec;
    std::filesystem::remove(m_db->m_path.str(), ec);

    std::filesystem::rename(tmpPath.str(), m_db->m_path.str());
  }

  return true;
}

denox::Db::Db(std::shared_ptr<DbMapped> db, std::shared_ptr<DbIndex> index)
    : m_db(std::move(db)), m_index(std::move(index)) {}

std::optional<denox::SpirvBinary>
denox::Db::query_shader_binary(const SHA256 &srcHash) const {
  auto it = m_index->binary_index.find(srcHash);
  if (it == m_index->binary_index.end()) {
    return std::nullopt;
  } else {
    uint32_t binaryId = static_cast<uint32_t>(it->second);
    return m_db->binaries[binaryId].spvBinary;
  }
}

bool denox::Db::insert_binary(const SHA256 &srcHash,
                              const SpirvBinary &binary) {
  auto cached = query_shader_binary(srcHash);
  if (cached) {
    if (!std::ranges::equal(binary.spv, cached->spv)) {
      DENOX_ERROR("Failed to insert binary into database: SHA256 collision in "
                  "database.");
      diag::invalid_state();
    }
    return false;
  }
  uint32_t binaryId = static_cast<uint32_t>(m_db->binaries.size());
  m_index->binary_index.emplace(srcHash, binaryId);
  m_db->binaries.emplace_back(DbShaderBinary{
      .hash = srcHash,
      .spvBinary = binary,
  });
  return true;
}

std::optional<std::chrono::duration<float, std::milli>>
denox::Db::query_dispatch_latency(const SHA256 &srcHash,
                                  std::span<const uint8_t> pushConstant,
                                  uint32_t workgroupCountX,
                                  uint32_t workgroupCountY,
                                  uint32_t workgroupCountZ) const {
  uint64_t hash = std::hash<SHA256>{}(srcHash);
  for (uint8_t b : pushConstant) {
    hash = algorithm::hash_combine(hash, b);
  }
  hash = algorithm::hash_combine(hash, workgroupCountX);
  hash = algorithm::hash_combine(hash, workgroupCountY);
  hash = algorithm::hash_combine(hash, workgroupCountZ);

  auto it = m_index->dispatch_buckets.find(hash);
  if (it == m_index->dispatch_buckets.end()) {
    return std::nullopt;
  }

  const auto &bucket = it->second;

  // linear search
  for (size_t dispatch_index : bucket) {
    const auto &dispatch = m_db->dispatches[dispatch_index];
    if (dispatch.workgroupCountX != workgroupCountX) {
      continue;
    }
    if (dispatch.workgroupCountY != workgroupCountY) {
      continue;
    }
    if (dispatch.workgroupCountZ != workgroupCountZ) {
      continue;
    }
    if (dispatch.pushConstant.size() != pushConstant.size()) {
      continue;
    }
    if (std::memcmp(dispatch.pushConstant.data(), pushConstant.data(),
                    dispatch.pushConstant.size()) != 0) {
      continue;
    }
    const auto &binary = m_db->binaries[dispatch.binaryId];
    if (binary.hash != srcHash) {
      continue;
    }
    if (!dispatch.time.has_value()) {
      return std::nullopt;
    }
    if (dispatch.time->samples.size() == 0) {
      return std::nullopt;
    }
    std::chrono::duration<uint64_t, std::nano> ns(
        dispatch.time->mean_latency_ns);
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
        ns);
  }
  return std::nullopt;
}

bool denox::Db::insert_dispatch(
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
    memory::optional<std::span<const uint32_t>> output_bindings) {
  uint32_t binaryId;
  {
    auto it = m_index->binary_index.find(srcHash);
    if (it == m_index->binary_index.end()) {
      DbShaderBinary binary;
      binary.hash = srcHash;
      binary.spvBinary = spvBinary;
      binaryId = static_cast<uint32_t>(m_db->binaries.size());
      m_db->binaries.push_back(binary);
      m_index->binary_index[srcHash] = binaryId;
    } else {
      binaryId = static_cast<uint32_t>(it->second);
      const auto &existing = m_db->binaries[binaryId].spvBinary;

      if (existing.spv.size() != spvBinary.spv.size()) {
        DENOX_ERROR("SPIR-V mismatch for identical shader hash! (different "
                    "binary-size)");
        diag::invalid_state();
      }
      if (std::memcmp(existing.spv.data(), spvBinary.spv.data(),
                      sizeof(uint32_t) * spvBinary.spv.size()) != 0) {
        DENOX_ERROR("SPIR-V mismatch for identical shader hash!");
        diag::invalid_state();
      }
    }
  }

  const auto &binary = m_db->binaries[binaryId];

  uint64_t hash = std::hash<SHA256>{}(binary.hash);
  for (uint8_t b : pushConstant) {
    hash = algorithm::hash_combine(hash, b);
  }
  hash = algorithm::hash_combine(hash, workgroupCountX);
  hash = algorithm::hash_combine(hash, workgroupCountY);
  hash = algorithm::hash_combine(hash, workgroupCountZ);

  auto it = m_index->dispatch_buckets.find(hash);
  if (it != m_index->dispatch_buckets.end()) {
    const auto &bucket = it->second;
    // linear search
    for (size_t dispatch_index : bucket) {
      const auto &dispatch = m_db->dispatches[dispatch_index];
      if (dispatch.workgroupCountX != workgroupCountX) {
        continue;
      }
      if (dispatch.workgroupCountY != workgroupCountY) {
        continue;
      }
      if (dispatch.workgroupCountZ != workgroupCountZ) {
        continue;
      }
      if (dispatch.pushConstant.size() != pushConstant.size()) {
        continue;
      }
      if (std::memcmp(dispatch.pushConstant.data(), pushConstant.data(),
                      dispatch.pushConstant.size()) != 0) {
        continue;
      }
      const auto &binary = m_db->binaries[dispatch.binaryId];
      if (binary.hash != srcHash) {
        continue;
      }
      return false;
    }
  }
  DbComputeDispatch dispatch;
  dispatch.binaryId = binaryId;
  dispatch.workgroupCountX = workgroupCountX;
  dispatch.workgroupCountY = workgroupCountY;
  dispatch.workgroupCountZ = workgroupCountZ;
  dispatch.pushConstant.assign(pushConstant.begin(), pushConstant.end());
  dispatch.hash = hash;
  dispatch.bindings.assign(bindings.begin(), bindings.end());
  dispatch.time = std::nullopt;
  dispatch.operation = operation;
  dispatch.shader_name = shader_name;
  dispatch.config = config;
  dispatch.memory_reads = memory_reads;
  dispatch.memory_writes = memory_writes;
  dispatch.flops = flops;
  dispatch.coopmat = coopmat.value_or(false);
  if (input_bindings) {
    dispatch.input_bindings.emplace(input_bindings->begin(),
                                    input_bindings->end());
  }
  if (output_bindings) {
    dispatch.output_bindings.emplace(output_bindings->begin(),
                                     output_bindings->end());
  }
  uint32_t dispatchIndex = static_cast<uint32_t>(m_db->dispatches.size());
  m_db->dispatches.push_back(std::move(dispatch));
  m_index->dispatch_buckets[hash].emplace_back(dispatchIndex);
  return true;
}

std::span<const denox::DbShaderBinary> denox::Db::binaries() const {
  return m_db->binaries;
}

std::span<const denox::DbComputeDispatch> denox::Db::dispatches() const {
  return m_db->dispatches;
}

void denox::Db::add_dispatch_benchmark_result(uint32_t dispatch_index,
                                              std::vector<DbSample> samples) {
  if (samples.empty()) {
    return;
  }

  assert(dispatch_index < m_db->dispatches.size());
  auto &dispatch = m_db->dispatches[dispatch_index];
  for (const DbSample &sample : samples) {
    if (!dispatch.time.has_value()) {
      dispatch.time.emplace();
      dispatch.time->samples.push_back(sample);
      dispatch.time->mean_latency_ns = sample.latency_ns;
      dispatch.time->std_derivation_ns = 0;
      continue;
    }
    const double prev_n = static_cast<double>(dispatch.time->samples.size());
    const double prev_mean_ns =
        static_cast<double>(dispatch.time->mean_latency_ns);
    const double prev_std_ns =
        static_cast<double>(dispatch.time->std_derivation_ns);

    const double latency_ns = static_cast<double>(sample.latency_ns);

    const double new_n = static_cast<double>(dispatch.time->samples.size() + 1);
    const double new_mean_ns = (prev_n * prev_mean_ns + 1 * latency_ns) / new_n;
    const double new_var_ns =
        std::max(0.0, (prev_n * (std::pow(prev_std_ns, 2) +
                                 std::pow(prev_mean_ns - new_mean_ns, 2)) +
                       std::pow(latency_ns - new_mean_ns, 2)) /
                          new_n);
    const double new_std_ns = std::sqrt(new_var_ns);

    dispatch.time->samples.push_back(sample);
    dispatch.time->mean_latency_ns = static_cast<uint64_t>(new_mean_ns);
    dispatch.time->std_derivation_ns = static_cast<uint64_t>(new_std_ns);
  }
}
const denox::io::Path &denox::Db::path() const { return m_db->m_path; }

uint32_t denox::Db::create_bench_environment(
    std::string device, std::string os, std::string driver_version,
    std::string denox_version, std::string denox_commit_hash,
    uint64_t start_timestamp, DbClockMode clockMode,
    uint16_t l2_warmup_iterations, uint16_t jit_warmup_iterations,
    uint16_t measurement_iterations) {
  uint32_t id = static_cast<uint32_t>(m_db->environments.size());
  m_db->environments.push_back(DbEnv{
      .device = device,
      .os = os,
      .driver_version = driver_version,
      .denox_version = denox_version,
      .denox_commit_hash = denox_commit_hash,
      .start_timestamp = start_timestamp,
      .clock_mode = clockMode,
      .l2_warmup_iterations = l2_warmup_iterations,
      .jit_warmup_iterations = jit_warmup_iterations,
      .measurement_iterations = measurement_iterations,
  });
  return id;
}

std::span<const denox::DbEnv> denox::Db::envs() const {
  return m_db->environments;
}
