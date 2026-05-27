#include "denox/cli/populate.hpp"
#include "denox/cli/device_yml/device_yml.hpp"
#include "denox/cli/io/InputStream.hpp"
#include "denox/compiler/populate.hpp"
#include "denox/db/Db.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"

void populate(PopulateAction &action) {
  denox::diag::Logger logger("denox.populate", action.logcolors,
                             action.loglevel);

  denox::memory::optional<denox::memory::string> deviceName;
  if (std::holds_alternative<IOEndpoint>(action.device)) {
    InputStream istream{std::get<IOEndpoint>(action.device)};
    denox::memory::vector<std::byte> yml = istream.read_all();
    action.options.deviceInfo = deserialize_device_yml(
        yml);
  } else if (std::holds_alternative<denox::memory::string>(action.device)) {
    deviceName = std::get<denox::memory::string>(action.device);
    action.options.deviceInfo =
        denox::query_driver_device_info(action.apiVersion, deviceName);
  } else {
    action.options.deviceInfo = denox::query_driver_device_info(
        action.apiVersion, denox::memory::nullopt);
  }

  auto db = denox::Db::open(action.database.endpoint.path());

  denox::populate(db, action.model.data, action.options, logger);

  db.checkpoint();
}
