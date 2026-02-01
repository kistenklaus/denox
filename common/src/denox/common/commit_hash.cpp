#include "denox/common/commit_hash.hpp"
#include "denox_commit_hash.hpp"

std::string denox::commit_hash() { return DENOX_GIT_COMMIT_HASH; }
