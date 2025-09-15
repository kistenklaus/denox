#include "memory/allocator/monotone_pool_allocator.hpp"
#include "ProfileAllocator.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <random>

using namespace denox;

TEST(memory_allocator, trivial_constructor_and_destructor) {
  {
    memory::monotonic_pool_allocator<sizeof(int), alignof(int)> pool(0);
  }
  {
    memory::monotonic_pool_allocator<sizeof(int), alignof(int)> pool(2);
  }
}

TEST(memory_allocator, alloc_dealloc_single) {
  denox::testing::ProfileAllocator profile{};
  struct Big {
    std::size_t v;
    std::uint64_t x[10];
  };
  {
    // NOTE: ChunkSize 2 => capacity = 1.
    memory::monotonic_pool_allocator<sizeof(Big), alignof(Big),
                                     denox::testing::ProfileAllocator>
        pool{2, profile};
    for (std::size_t i = 0; i < 10000; ++i) {
      void *ptr = pool.allocate(sizeof(Big), alignof(Big));
      EXPECT_TRUE(ptr != nullptr);
      static_cast<volatile Big *>(ptr)->v = i;
      pool.deallocate(ptr);
    }
    EXPECT_EQ(profile.peakAllocationCount(), 1);
    EXPECT_EQ(profile.peakAllocatedBytes(), 2 * sizeof(Big));
  }
  EXPECT_EQ(profile.allocationCount(), 0);
  EXPECT_EQ(profile.allocatedBytes(), 0);
}

TEST(memory_allocator, alloc_dealloc_permutation) {
  denox::testing::ProfileAllocator profile{};
  struct Big {
    std::size_t v;
    std::uint64_t x[10];
  };
  {
    static constexpr std::size_t PEAK = 1000;
    static_assert(PEAK % 2 == 0);
    memory::monotonic_pool_allocator<sizeof(Big), alignof(Big),
                                     denox::testing::ProfileAllocator,
                                     std::ratio<1, 1>>
        pool{PEAK / 2 + 1, profile};

    std::random_device rng;
    std::mt19937 prng(rng());
    for (std::size_t it = 0; it < 10; ++it) {

      std::vector<Big *> bigs(PEAK);
      std::vector<std::size_t> perm(PEAK);
      for (std::size_t i = 0; i < PEAK; ++i) {
        bigs[i] = static_cast<Big *>(pool.allocate(sizeof(Big), alignof(Big)));
        bigs[i]->v = i;
        perm[i] = i;
      }
      std::shuffle(perm.begin(), perm.end(), prng);
      for (std::size_t i = 0; i < PEAK; ++i) {
        std::size_t j = perm[i];
        bigs[j]->v = j;
        pool.deallocate(static_cast<void *>(bigs[j]));
      }
    }
    EXPECT_EQ(profile.peakAllocationCount(), 2);
    EXPECT_EQ(profile.peakAllocationCount(), profile.allocationCount());
    // NOTE: 2 * Big is the header of each block PEAK * Big is the actually
    // managed memory.
    EXPECT_EQ(profile.peakAllocatedBytes(),
              sizeof(Big) * PEAK + 2 * sizeof(Big));
    EXPECT_EQ(profile.peakAllocatedBytes(), profile.allocatedBytes());
  }

  EXPECT_EQ(profile.allocationCount(), 0);
  EXPECT_EQ(profile.allocatedBytes(), 0);
}
