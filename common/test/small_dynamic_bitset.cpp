#include "denox/memory/container/small_dynamic_bitset.hpp"
#include <gtest/gtest.h>
#include <utility>

using Bits = denox::memory::small_dynamic_bitset<64>;

TEST(SmallDynamicBitset, ShrinkAndRegrowReusesStorageWithoutRestoringOldBits) {
  Bits bits(257, true);
  const auto *storage = bits.words();
  for (size_t size : {193u, 65u, 63u, 0u}) {
    bits.resize(size);
    EXPECT_EQ(bits.words(), storage);
    EXPECT_EQ(bits.count(), size);
    bits.resize(257);
    EXPECT_EQ(bits.words(), storage);
    EXPECT_EQ(bits.count(), size);
    for (size_t i = size; i < bits.size(); ++i) EXPECT_FALSE(bits[i]);
    bits.set_all();
  }
}

TEST(SmallDynamicBitset, GrowthInitializesBothPartialAndWholeWords) {
  Bits bits(63, true);
  bits.resize(65);
  EXPECT_EQ(bits.count(), 63u);
  EXPECT_FALSE(bits[63]);
  EXPECT_FALSE(bits[64]);
  bits.resize(257, true);
  EXPECT_EQ(bits.count(), 255u);
  bits.resize(3);
  bits.resize(257, true);
  EXPECT_EQ(bits.count(), 257u);
  EXPECT_EQ(bits.words()[4], 1u); // Unused tail bits remain clear.
}

TEST(SmallDynamicBitset, CopyAndMoveAfterShrinkingHeapStorage) {
  Bits bits(257, true);
  bits.resize(3);
  const auto *storage = bits.words();
  Bits copy = bits;
  EXPECT_EQ(copy, bits);
  Bits moved = std::move(bits);
  EXPECT_EQ(moved, copy);
  EXPECT_EQ(moved.words(), storage);
  EXPECT_TRUE(bits.empty());
  moved.resize(257);
  EXPECT_EQ(moved.words(), storage);
  EXPECT_EQ(moved.count(), 3u);
  bits.resize(65);
  EXPECT_TRUE(bits.none());
  copy = std::move(moved);
  EXPECT_EQ(copy.words(), storage);
  EXPECT_EQ(copy.count(), 3u);
  moved.resize(257);
  EXPECT_TRUE(moved.none());
}

TEST(SmallDynamicBitset, ZeroInlineCapacityAndClearRegrowth) {
  denox::memory::small_dynamic_bitset<0> bits(65, true);
  const auto *storage = bits.words();
  bits.clear();
  bits.resize(65);
  EXPECT_EQ(bits.words(), storage);
  EXPECT_TRUE(bits.none());
}
