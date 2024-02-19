
#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

#include "gtest/gtest.h"

TEST(IsLeapYear, Base) {
  EXPECT_TRUE(IsLeapYear(1972));
  EXPECT_FALSE(IsLeapYear(1970));
}

TEST(UTCMkTime, Base) {
  // Tested with www.epochconverter.com
  EXPECT_EQ(UTCMkTime(1970, 1, 1, 0, 0, 0).value().seconds_since_epoch, 0);
  EXPECT_EQ(UTCMkTime(1970, 1, 1, 0, 0, 0).value().wday, 4);  // Thursday

  EXPECT_EQ(UTCMkTime(2024, 2, 19, 10, 8, 55).value().seconds_since_epoch,
            1708337335);
  EXPECT_EQ(UTCMkTime(2024, 2, 19, 10, 8, 55).value().wday, 1);  // Monday
}
