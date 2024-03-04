#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

#include <cstdint>
#include <optional>

// Number of days in each month (non-leap year)
constexpr int daysPerMonth[12] = {31, 28, 31, 30, 31, 30,
                                  31, 31, 30, 31, 30, 31};

bool IsLeapYear(const int year) {
  return ((year % 4) == 0) && ((year % 100) != 0 || (year % 400) == 0);
}

std::optional<MyTime> UTCMkTime(const int year, const int month, const int day,
                                const int hour, const int minute,
                                const int second) {
  const bool is_leap_year = IsLeapYear(year);

  if (month < 1 || month > 12) {
    // Invalid date.
    return {};
  }

  if ((day > daysPerMonth[month - 1]) &&
      !(is_leap_year && month == 2 && day == 29)) {
    // Invalid date.
    return {};
  }

  constexpr int64_t seconds_per_day = 24 * 60 * 60;

  // Seconds since Unix Epoch to start of the year.
  int64_t seconds_since_epoch = (year - 1970) * 365 * seconds_per_day;

  // Add extra days for leap years.
  if (year >= 1970) {
    for (int cur_year = 1972; cur_year < year; cur_year += 4) {
      if (IsLeapYear(cur_year)) {
        seconds_since_epoch += seconds_per_day;
      }
    }

    // Has February 29 days?
    if (is_leap_year && month > 2) {
      seconds_since_epoch += seconds_per_day;
    }
  } else {
    for (int cur_year = 1968; cur_year > year; cur_year -= 4) {
      if (IsLeapYear(cur_year)) {
        seconds_since_epoch -= seconds_per_day;
      }
    }

    // Has February 29 days?
    if (is_leap_year && month <= 2) {
      seconds_since_epoch -= seconds_per_day;
    }
  }

  // Month days.
  for (int cur_month = 0; cur_month < month - 1; cur_month++) {
    seconds_since_epoch += daysPerMonth[cur_month] * seconds_per_day;
  }

  seconds_since_epoch += (day - 1) * seconds_per_day;
  seconds_since_epoch += hour * 60 * 60;
  seconds_since_epoch += minute * 60;
  seconds_since_epoch += second;

  const auto days_since_epoch = seconds_since_epoch / seconds_per_day;
  // Note: 1970-1-1 was a Thursday.
  int week_days = static_cast<int>((days_since_epoch + 4) % 7);
  if (week_days < 0) {
    week_days += 7;
  }

  return MyTime{.seconds_since_epoch = seconds_since_epoch, .wday = week_days};
}
