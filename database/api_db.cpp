#include "db/Db.hpp"
#include "denox/compiler.h"
#include "diag/logging.hpp"
#include "io/fs/Path.hpp"

namespace denox::api {
class Db {
public:
  static constexpr uint32_t MAGIC = 0x1287361;
  uint32_t magic;
  denox::compiler::Db db;
};
}; // namespace denox::api

DenoxResult denox_open_database(const char *cpath, DenoxDB *db) {
  if (cpath == nullptr) {
    DENOX_ERROR("Failed to open database: path = NULL");
    return DenoxResult::DENOX_FAILURE;
  }
  if (db == nullptr) {
    DENOX_ERROR("Failed to open database: db = NULL");
    return DenoxResult::DENOX_FAILURE;
  }
  const denox::io::Path path{cpath};

  try {
    auto database = denox::compiler::Db::open(path);
    denox::api::Db *apidb =
        new denox::api::Db(denox::api::Db::MAGIC, std::move(database));
    *db = static_cast<DenoxDB>(apidb);
  } catch (const std::exception &e) {
    return DENOX_FAILURE;
  }
}

DenoxResult denox_close_database(DenoxDB db) {
  if (db == nullptr) {
    DENOX_ERROR("Failed to close database: db is not open");
    return DenoxResult::DENOX_FAILURE;
  }
  auto *apidb = static_cast<denox::api::Db *>(db);
  if (apidb->magic != denox::api::Db::MAGIC) {
    DENOX_ERROR("Failed to close database: invalid database");
    return DenoxResult::DENOX_FAILURE;
  }
  apidb->db.write_back();
}
