#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

#include <cstdint>
#include <optional>

// Number of days in each month (non-leap year)
constexpr int daysPerMonth[12] = {31, 28, 31, 30, 31, 30,
                                  31, 31, 30, 31, 30, 31};

bool IsLeapYear(int year) {
  return ((year % 4) == 0) && ((year % 100) != 0 || (year % 400) == 0);
}

std::optional<MyTime> UTCMkTime(const int year, const int month, const int day,
                                const int hour, const int minute,
                                const int second) {
  if (month < 1 || month > 12 || day > daysPerMonth[month - 1]) {
    // Invalid date.
    return {};
  }

  const int64_t seconds_per_days = 24 * 60 * 60;

  // Seconds since Unix Epoch to start of the year.
  int64_t seconds_since_epoch = (year - 1970) * 365 * seconds_per_days;

  // Add extra days for leap years.
  for (int cur_year = 1972; cur_year < year; cur_year += 4) {
    if (IsLeapYear(cur_year)) {
      seconds_since_epoch += seconds_per_days;
    }
  }

  // Month days.
  for (int cur_month = 0; cur_month < month - 1; cur_month++) {
    seconds_since_epoch += daysPerMonth[cur_month] * seconds_per_days;
  }

  // Has February 29 days?
  if (IsLeapYear(year) && month > 2) {
    seconds_since_epoch += seconds_per_days;
  }

  seconds_since_epoch += (day - 1) * seconds_per_days;
  seconds_since_epoch += hour * 60 * 60;
  seconds_since_epoch += minute * 60;
  seconds_since_epoch += second;

  const auto days_since_epoch = seconds_since_epoch / seconds_per_days;
  // Note: 1970-1-1 was a thursday.
  const int week_days = static_cast<int>((days_since_epoch + 4) % 7);

  return MyTime{.seconds_since_epoch = seconds_since_epoch, .wday = week_days};
}
