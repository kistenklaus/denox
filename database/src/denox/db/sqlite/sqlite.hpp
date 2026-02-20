#pragma once

#include "denox/diag/logging.hpp"
#include "denox/io/fs/FileLock.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"

#include <cstring>
#include <span>
#include <sqlite3.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace denox::sqlite {

namespace details {

[[noreturn]] inline void sqlite_error(std::string_view what, int rc,
                                      const char *msg) {
  if (msg) {
    throw std::runtime_error(
        fmt::format("SQLite error: {} (rc={}, msg={})", what, rc, msg));
  } else {
    throw std::runtime_error(fmt::format("SQLite error: {} (rc={})", what, rc));
  }
}

} // namespace details

struct Stmt {
public:
  Stmt() : m_stmt(nullptr) {}
  Stmt(const Stmt &) = delete;
  Stmt &operator=(const Stmt &) = delete;

  Stmt(Stmt &&o) noexcept : m_stmt(std::exchange(o.m_stmt, nullptr)) {}

  Stmt &operator=(Stmt &&o) noexcept {
    if (this != &o) {
      finalize();
      m_stmt = std::exchange(o.m_stmt, nullptr);
    }
    return *this;
  }

  ~Stmt() { finalize(); }

  void bind_int(int idx, int v) {
    int rc = sqlite3_bind_int(m_stmt, idx, v);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_int", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_int64(int idx, int64_t v) {
    int rc = sqlite3_bind_int64(m_stmt, idx, v);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_int64", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_double(int idx, double v) {
    int rc = sqlite3_bind_double(m_stmt, idx, v);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_double", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_cstr(int idx, const char *v) {
    int rc = sqlite3_bind_text(m_stmt, idx, v, -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_text", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_sv(int idx, std::string_view v) {
    int rc = sqlite3_bind_text(m_stmt, idx, v.data(),
                               static_cast<int>(v.size()), SQLITE_TRANSIENT);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_text", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_blob(int idx, const void *data, int size) {
    int rc = sqlite3_bind_blob(m_stmt, idx, data, size, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_blob", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void bind_null(int idx) {
    int rc = sqlite3_bind_null(m_stmt, idx);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_bind_null", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  void clear_bindings() { sqlite3_clear_bindings(m_stmt); }

  bool next() {
    int rc = sqlite3_step(m_stmt);
    if (rc == SQLITE_ROW)
      return true;
    if (rc == SQLITE_DONE)
      return false;
    details::sqlite_error("sqlite3_step", rc,
                          sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
    return false;
  }

  void reset() {
    int rc = sqlite3_reset(m_stmt);
    if (rc != SQLITE_OK)
      details::sqlite_error("sqlite3_reset", rc,
                            sqlite3_errmsg(sqlite3_db_handle(m_stmt)));
  }

  int column_count() const { return sqlite3_column_count(m_stmt); }

  bool is_null(int col) const {
    return sqlite3_column_type(m_stmt, col) == SQLITE_NULL;
  }

  int as_int(int col) const {
    if (is_null(col))
      details::sqlite_error("as_int: NULL column", SQLITE_MISUSE, nullptr);
    return sqlite3_column_int(m_stmt, col);
  }

  int64_t as_int64(int col) const {
    if (is_null(col))
      details::sqlite_error("as_int64: NULL column", SQLITE_MISUSE, nullptr);
    return sqlite3_column_int64(m_stmt, col);
  }

  memory::optional<int> as_optional_int(int col) const {
    if (is_null(col))
      return memory::nullopt;
    return sqlite3_column_int(m_stmt, col);
  }

  memory::optional<int64_t> as_optional_int64(int col) const {
    if (is_null(col))
      return memory::nullopt;
    return sqlite3_column_int64(m_stmt, col);
  }

  double as_double(int col) const {
    if (is_null(col))
      details::sqlite_error("as_double: NULL column", SQLITE_MISUSE, nullptr);
    return sqlite3_column_double(m_stmt, col);
  }

  memory::optional<double> as_optional_double(int col) const {
    if (is_null(col))
      return memory::nullopt;
    return sqlite3_column_double(m_stmt, col);
  }

  float as_float(int col) const {
    if (is_null(col))
      details::sqlite_error("as_float: NULL column", SQLITE_MISUSE, nullptr);
    return static_cast<float>(sqlite3_column_double(m_stmt, col));
  }

  memory::optional<float> as_optional_float(int col) const {
    if (is_null(col))
      return memory::nullopt;
    return static_cast<float>(sqlite3_column_double(m_stmt, col));
  }

  memory::string as_string(int col) const {
    if (is_null(col))
      details::sqlite_error("as_string: NULL column", SQLITE_MISUSE, nullptr);

    const unsigned char *ptr = sqlite3_column_text(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return memory::string(reinterpret_cast<const char *>(ptr),
                          static_cast<size_t>(len));
  }

  memory::optional<memory::string> as_optional_string(int col) const {
    if (is_null(col))
      return memory::nullopt;

    const unsigned char *ptr = sqlite3_column_text(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return memory::string(reinterpret_cast<const char *>(ptr),
                          static_cast<size_t>(len));
  }

  std::string_view as_string_view(int col) const {
    if (is_null(col))
      details::sqlite_error("as_string_view: NULL column", SQLITE_MISUSE,
                            nullptr);

    const unsigned char *ptr = sqlite3_column_text(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return std::string_view(reinterpret_cast<const char *>(ptr),
                            static_cast<size_t>(len));
  }

  memory::optional<std::string_view> as_optional_string_view(int col) const {
    if (is_null(col))
      return memory::nullopt;

    const unsigned char *ptr = sqlite3_column_text(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return std::string_view(reinterpret_cast<const char *>(ptr),
                            static_cast<size_t>(len));
  }

  const char *as_cstr(int col) const {
    if (is_null(col))
      details::sqlite_error("as_cstr: NULL column", SQLITE_MISUSE, nullptr);

    return reinterpret_cast<const char *>(sqlite3_column_text(m_stmt, col));
  }

  memory::optional<const char *> as_optional_cstr(int col) const {
    if (is_null(col))
      return memory::nullopt;

    return reinterpret_cast<const char *>(sqlite3_column_text(m_stmt, col));
  }

  std::span<const std::byte> as_blob_view(int col) const {
    if (is_null(col))
      details::sqlite_error("as_blob_view: NULL column", SQLITE_MISUSE,
                            nullptr);

    const void *ptr = sqlite3_column_blob(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return std::span<const std::byte>(reinterpret_cast<const std::byte *>(ptr),
                                      static_cast<size_t>(len));
  }

  memory::optional<std::span<const std::byte>>
  as_optional_blob_view(int col) const {
    if (is_null(col))
      return memory::nullopt;

    const void *ptr = sqlite3_column_blob(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    return std::span<const std::byte>(reinterpret_cast<const std::byte *>(ptr),
                                      static_cast<size_t>(len));
  }

  memory::vector<std::byte> as_blob(int col) const {
    if (is_null(col))
      details::sqlite_error("as_blob: NULL column", SQLITE_MISUSE, nullptr);

    const void *ptr = sqlite3_column_blob(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    memory::vector<std::byte> out(static_cast<size_t>(len));
    std::memcpy(out.data(), ptr, static_cast<size_t>(len));
    return out;
  }

  memory::optional<memory::vector<std::byte>> as_optional_blob(int col) const {
    if (is_null(col))
      return memory::nullopt;

    const void *ptr = sqlite3_column_blob(m_stmt, col);
    int len = sqlite3_column_bytes(m_stmt, col);

    memory::vector<std::byte> out(static_cast<size_t>(len));
    std::memcpy(out.data(), ptr, static_cast<size_t>(len));
    return out;
  }

  void finalize() {
    if (m_stmt) {
      sqlite3_finalize(m_stmt);
      m_stmt = nullptr;
    }
  }

private:
  explicit Stmt(sqlite3_stmt *stmt) : m_stmt(stmt) {}

  sqlite3_stmt *m_stmt = nullptr;

  friend class Db;
};

class Db {
public:
  Db() = default;
  Db(const Db &) = delete;
  Db &operator=(const Db &) = delete;

  Db(Db &&o) noexcept
      : m_path(std::move(o.m_path)), m_filelock(std::move(o.m_filelock)),
        m_handle(std::exchange(o.m_handle, nullptr)) {}

  Db &operator=(Db &&o) noexcept {
    if (this != &o) {
      close();
      m_path = std::move(o.m_path);
      m_filelock = std::move(o.m_filelock);
      m_handle = std::exchange(o.m_handle, nullptr);
    }
    return *this;
  }

  ~Db() { close(); }

  static Db open(io::Path path) {
    sqlite3 *handle = nullptr;
    io::FileLock lock;
    const bool in_memory = path.empty(); // or however you detect empty
    if (in_memory) {
      int rc = sqlite3_open_v2(":memory:", &handle,
                               SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                   SQLITE_OPEN_NOMUTEX,
                               nullptr);
      if (rc != SQLITE_OK) {
        const char *msg = handle ? sqlite3_errmsg(handle) : nullptr;
        if (handle)
          sqlite3_close_v2(handle);
        details::sqlite_error("open(:memory:)", rc, msg);
      }
    } else {
      io::Path lockpath(path.str() + ".lock");
      lock = io::FileLock::lock(lockpath);
      int rc =
          sqlite3_open_v2(path.cstr(), &handle,
                          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                              SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_PRIVATECACHE,
                          nullptr);
      if (rc != SQLITE_OK) {
        const char *msg = handle ? sqlite3_errmsg(handle) : nullptr;
        if (handle)
          sqlite3_close_v2(handle);
        details::sqlite_error("open(file)", rc, msg);
      }
    }
    Db db(path, std::move(lock), handle);

    if (!in_memory) {
      db.exec("PRAGMA journal_mode=WAL;");
      db.exec("PRAGMA synchronous=NORMAL;");
    }

    db.exec("PRAGMA foreign_keys=ON;");
    db.exec("PRAGMA busy_timeout=5000;");

    return db;
  }

  template <typename Fn> auto with_transaction(Fn &&fn) {
    using Ret = decltype(fn());
    exec("BEGIN IMMEDIATE;");
    try {
      if constexpr (std::is_void_v<Ret>) {
        fn();
        exec("COMMIT;");
        return;
      } else {
        Ret result = fn();
        exec("COMMIT;");
        return result;
      }
    } catch (...) {
      try {
        exec("ROLLBACK;");
      } catch (...) {
        // swallow rollback failure
      }
      throw;
    }
  }

  Stmt prepare(const char *sql) {
    sqlite3_stmt *stmt = nullptr;
    int rc = sqlite3_prepare_v2(m_handle, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK)
      details::sqlite_error(sql, rc, sqlite3_errmsg(m_handle));
    return Stmt(stmt);
  }

  void exec(const std::string &sql) { exec(sql.c_str()); }

  void exec(const char *sql) {
    char *err = nullptr;
    int rc = sqlite3_exec(m_handle, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
      const char *msg = err ? err : sqlite3_errmsg(m_handle);
      if (err)
        sqlite3_free(err);
      details::sqlite_error(sql, rc, msg);
    }
  }
  int64_t last_insert_rowid() const {
    return sqlite3_last_insert_rowid(m_handle);
  }

  void checkpoint() { exec("PRAGMA wal_checkpoint(TRUNCATE);"); }

  void close() {
    if (m_handle) {
      int rc = sqlite3_close_v2(m_handle);
      if (rc != SQLITE_OK)
        details::sqlite_error("sqlite3_close_v2", rc, sqlite3_errmsg(m_handle));
      m_handle = nullptr;
    }
  }

private:
  Db(io::Path path, io::FileLock lock, sqlite3 *handle)
      : m_path(std::move(path)), m_filelock(std::move(lock)), m_handle(handle) {
  }

  io::Path m_path;
  io::FileLock m_filelock;
  sqlite3 *m_handle = nullptr;
};

} // namespace denox::sqlite
