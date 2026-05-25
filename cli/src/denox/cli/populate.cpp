#include "denox/cli/populate.hpp"
#include "denox/compiler/populate.hpp"
#include "denox/db/Db.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"

void populate(PopulateAction &action) {
  denox::diag::Logger logger("denox.populate", action.logcolors, action.loglevel);

  denox::memory::optional<denox::memory::string> deviceName;
  if (std::holds_alternative<IOEndpoint>(action.device)) {
    throw std::runtime_error("invalid device");
  } else if (std::holds_alternative<denox::memory::string>(action.device)) {
    deviceName = std::get<denox::memory::string>(action.device);
  }

  denox::ApiVersion apiVersion = action.apiVersion;
  action.options.deviceInfo =
      denox::query_driver_device_info(apiVersion, deviceName);

  auto db = denox::Db::open(action.database.endpoint.path());

  denox::populate(db, action.model.data, action.options, logger);

  db.checkpoint();
}
