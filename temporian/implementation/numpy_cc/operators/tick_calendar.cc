#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

// Number of days in each month (non-leap year)
constexpr int daysPerMonth[12] = {31, 28, 31, 30, 31, 30,
                                  31, 31, 30, 31, 30, 31};

bool IsLeapYear(int year) {
  return ((year % 4) == 0) && ((year % 100) != 0 || (year % 400) == 0);
}

struct MyTime {
  int64_t seconds_since_epoch;
  int wday;
};

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

py::array_t<double> tick_calendar(
    const double start_timestamp,                // min date
    const double end_timestamp,                  // max date
    const int min_second, const int max_second,  // second range
    const int min_minute, const int max_minute,  // minute range
    const int min_hour, const int max_hour,      // hours range
    const int min_mday, const int max_mday,      // month days
    const int min_month, const int max_month,    // month range
    const int min_wday, const int max_wday       // weekdays
) {
  // Ticks list
  std::vector<double> ticks;

  // Date range
  const auto start_t = static_cast<std::time_t>(std::floor(start_timestamp));
  const auto end_t = static_cast<std::time_t>(std::floor(end_timestamp));

  std::tm start_utc = *std::gmtime(&start_t);

  int year = start_utc.tm_year + 1900;
  int month = std::max(start_utc.tm_mon + 1, min_month);  // 1-12
  int mday = std::max(start_utc.tm_mday, min_mday);       // 1-31
  int hour = std::max(start_utc.tm_hour, min_hour);
  int minute = std::max(start_utc.tm_min, min_minute);
  int second = std::max(start_utc.tm_sec, min_second);

  bool in_range = true;
  while (in_range) {
    while (month <= max_month && in_range) {
      while (mday <= max_mday && in_range) {
        while (hour <= max_hour && in_range) {
          while (minute <= max_minute && in_range) {
            while (second <= max_second && in_range) {
              const auto cur_time_or =
                  UTCMkTime(year, month, mday, hour, minute, second);

              // Valid date
              if (cur_time_or.has_value()) {
                const auto cur_time = cur_time_or.value();

                // Finish condition
                if (cur_time.seconds_since_epoch > end_t) {
                  in_range = false;
                  break;
                }

                // Check weekday match
                if (cur_time.wday >= min_wday && cur_time.wday <= max_wday) {
                  ticks.push_back(cur_time.seconds_since_epoch);
                }
              } else {
                // Invalid date (e.g: 31/4)
                second = max_second;  // avoid unnecessary loops
                minute = max_minute;
                hour = max_hour;
              }
              second++;
            }
            second = min_second;
            minute++;
          }
          second = min_second;
          minute = min_minute;
          hour++;
        }
        second = min_second;
        minute = min_minute;
        hour = min_hour;
        mday++;
      }
      second = min_second;
      minute = min_minute;
      hour = min_hour;
      mday = min_mday;
      month++;
    }
    second = min_second;
    minute = min_minute;
    hour = min_hour;
    mday = min_mday;
    month = min_month;
    year++;
  }
  // TODO: optimize mday += 7 on specific wdays

  // Allocate output array
  // TODO: can we avoid this data copy?
  py::array_t<double> result(ticks.size());
  std::copy(ticks.begin(), ticks.end(), result.mutable_data());
  return result;
}

}  // namespace

void init_tick_calendar(py::module& m) {
  m.def("tick_calendar", &tick_calendar, "", py::arg("start_timestamp"),
        py::arg("end_timestamp"), py::arg("min_second"), py::arg("max_second"),
        py::arg("min_minute"), py::arg("max_minute"), py::arg("min_hour"),
        py::arg("max_hour"), py::arg("min_mday"), py::arg("max_mday"),
        py::arg("min_month"), py::arg("max_month"), py::arg("min_wday"),
        py::arg("max_wday"));
}
