#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <optional>

#include "absl/time/time.h"

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

namespace py = pybind11;
const int SECONDS_PER_HOURS = 60 * 60;

ErrorString parse_tz(const py::object &tz, absl::TimeZone &parsed_tz) {
  if (py::isinstance<py::int_>(tz)) {
    // handle tz as an int
    const int int_tz = tz.cast<int>();
    // tz is expresed in hours, needs to be converted to seconds
    parsed_tz = absl::FixedTimeZone(int_tz * SECONDS_PER_HOURS);

  } else if (py::isinstance<py::float_>(tz)) {
    // handle tz as an float
    const float float_tz = tz.cast<float>() * SECONDS_PER_HOURS;
    const int int_tz = static_cast<int>(std::round(float_tz));
    // tz is expresed in hours, needs to be converted to seconds
    parsed_tz = absl::FixedTimeZone(int_tz);

  } else if (py::isinstance<py::str>(tz)) {
    // handle tz as an string
    const std::string str_tz = tz.cast<std::string>();
    if (!absl::LoadTimeZone(str_tz, &parsed_tz)) {
      return "Invalid timezone '" + str_tz +
             "'. Only names defined in the IANA timezone" +
             "database are valid";
    }

  } else {
    return "Unsupported argument type for \"tz\" argument. Expecting int, "
           "float, "
           "or str";
  }

  return {};
}

// TODO: support week starting on Sunday
int map_week_day(const absl::Weekday &wd) {
  switch (wd) {
    case absl::Weekday::monday:
      return 0;
    case absl::Weekday::tuesday:
      return 1;
    case absl::Weekday::wednesday:
      return 2;
    case absl::Weekday::thursday:
      return 3;
    case absl::Weekday::friday:
      return 4;
    case absl::Weekday::saturday:
      return 5;
    case absl::Weekday::sunday:
      return 6;
  }
}