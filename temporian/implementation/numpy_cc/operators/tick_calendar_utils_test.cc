
#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

#include "gtest/gtest.h"

TEST(IsLeapYear, Base) {
  EXPECT_TRUE(IsLeapYear(1972));
  EXPECT_TRUE(IsLeapYear(2000));

  EXPECT_FALSE(IsLeapYear(1970));
  EXPECT_FALSE(IsLeapYear(1900));
}

struct UTCMkTimeTestCase {
  const int year;
  const int month;
  const int day;
  const int hour;
  const int minute;
  const int second;
  const int64_t expected_seconds_since_epoch;
  const int expected_wday;
};

using UTCMkTimeTest = testing::TestWithParam<UTCMkTimeTestCase>;

TEST_P(UTCMkTimeTest, MatchExpected) {
  const UTCMkTimeTestCase& test_case = GetParam();
  const auto result =
      UTCMkTime(test_case.year, test_case.month, test_case.day, test_case.hour,
                test_case.minute, test_case.second);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value().seconds_since_epoch,
            test_case.expected_seconds_since_epoch);
  EXPECT_EQ(result.value().wday, test_case.expected_wday);
}

INSTANTIATE_TEST_SUITE_P(UTCMkTimeTestBase, UTCMkTimeTest,
                         testing::ValuesIn<UTCMkTimeTestCase>({
                             {1900, 1, 1, 0, 0, 0, -2208988800, 1 /*Monday*/},

                             {1963, 1, 1, 0, 0, 0, -220924800, 2},
                             {1963, 2, 28, 0, 0, 0, -215913600, 4},
                             {1963, 3, 1, 0, 0, 0, -215827200, 5},

                             {1964, 1, 1, 0, 0, 0, -189388800, 3},
                             {1964, 2, 28, 0, 0, 0, -184377600, 5},
                             {1964, 2, 29, 0, 0, 0, -184291200, 6},
                             {1964, 3, 1, 0, 0, 0, -184204800, 0},

                             {1965, 1, 1, 0, 0, 0, -157766400, 5},
                             {1965, 2, 28, 0, 0, 0, -152755200, 0},
                             {1965, 3, 1, 0, 0, 0, -152668800, 1},

                             {1970, 1, 1, 0, 0, 0, 0, 4 /*Thursday*/},
                             {2024, 2, 19, 10, 8, 55, 1708337335, 1 /*Monday*/},
                             {2000, 2, 29, 0, 0, 0, 951782400, 2 /*Tuesday*/},
                             {3000, 1, 1, 0, 0, 0, 32503680000,
                              3 /*Wednesday*/},
                         }));

TEST_P(UTCMkTimeTest, IsInvalid) {
  EXPECT_FALSE(UTCMkTime(1900, 4, 31, 0, 0, 0).has_value());
  EXPECT_FALSE(UTCMkTime(1900, 2, 29, 0, 0, 0).has_value());
}

TEST(map_week_day, Base) { EXPECT_EQ(map_week_day(absl::Weekday::monday), 0); }

TEST(parse_tz, Base) {
  absl::TimeZone parsed_tz;
  auto error = parse_tz(py::str("invalid tz"), parsed_tz);
  EXPECT_TRUE(error.has_value());

  error = parse_tz(py::str("UTC"), parsed_tz);
  EXPECT_FALSE(error.has_value());

  error = parse_tz(py::str("US/Pacific"), parsed_tz);
  EXPECT_FALSE(error.has_value());
}