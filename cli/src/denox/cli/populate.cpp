#include "denox/cli/populate.hpp"
#include "denox/compiler/populate.hpp"
#include "denox/db/Db.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"

void populate(PopulateAction &action) {
  denox::ApiVersion apiVersion = action.apiVersion;
  action.options.deviceInfo =
      denox::query_driver_device_info(apiVersion, action.deviceName);

  auto db = denox::Db::open(action.database.endpoint.path());

  denox::populate(db, action.model.data, action.options);

  db.atomic_writeback();
}
