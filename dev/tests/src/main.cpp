#include <gtest/gtest.h>

#include "env.hpp"


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  env = new TestEnv{};
  ::testing::AddGlobalTestEnvironment(env);

  return RUN_ALL_TESTS();
}
