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
#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

namespace {
namespace py = pybind11;

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
                // NOTE: std:optional.value() fails to build on github runner
                // for arm64 macos although this function is on the
                // c++17 standard.
                // This alternative way to access the value works correctly.
                const auto cur_time = *cur_time_or;

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
