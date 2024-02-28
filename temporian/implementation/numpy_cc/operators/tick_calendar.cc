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

bool in_range(const int value, const int min, const int max) {
  return value >= min && value <= max;
}

std::vector<double> find_ticks(
    double start_timestamp, const std::optional<double> end_timestamp,
    const bool forward, const int max_ticks, const int min_second,
    const int max_second,                        // second range
    const int min_minute, const int max_minute,  // minute range
    const int min_hour, const int max_hour,      // hours range
    const int min_mday, const int max_mday,      // month days
    const int min_month, const int max_month,    // month range
    const int min_wday, const int max_wday       // weekdays
) {
  // Ticks list
  std::vector<double> ticks;

  // Direction in wich to search
  int step = forward ? 1 : -1;

  // End time
  const std::optional<std::time_t> end_t =
      end_timestamp.has_value()
          ? std::optional<std::time_t>{static_cast<std::time_t>(
                std::floor(end_timestamp.value()))}
          : std::nullopt;

  // Start time
  std::time_t start_t;
  std::tm start_utc;

  // The main loop needs the start_utc to be the first datetime >= to the
  // start_t that is contained in the ranges defined by all the min-max
  do {
    start_t = static_cast<std::time_t>(std::floor(start_timestamp));
    start_utc = *std::gmtime(&start_t);
    start_timestamp += step;
  } while (!(in_range(start_utc.tm_mon + 1, min_month, max_month) &&
             in_range(start_utc.tm_mday, min_mday, max_mday) &&
             in_range(start_utc.tm_hour, min_hour, max_hour) &&
             in_range(start_utc.tm_min, min_minute, max_minute) &&
             in_range(start_utc.tm_sec, min_second, max_second) &&
             in_range(start_utc.tm_wday, min_wday, max_wday)));

  int year = start_utc.tm_year + 1900;
  int month = start_utc.tm_mon + 1;  // zero-based tm_mon
  int mday = start_utc.tm_mday;
  int hour = start_utc.tm_hour;
  int minute = start_utc.tm_min;
  int second = start_utc.tm_sec;

  bool keep_looking = true;
  while (keep_looking) {
    while (in_range(month, min_month, max_month) && keep_looking) {
      while (in_range(mday, min_mday, max_mday) && keep_looking) {
        while (in_range(hour, min_hour, max_hour) && keep_looking) {
          while (in_range(minute, min_minute, max_minute) && keep_looking) {
            while (in_range(second, min_second, max_second) && keep_looking) {
              const auto cur_time_or =
                  UTCMkTime(year, month, mday, hour, minute, second);

              // Valid date
              if (cur_time_or.has_value()) {
                // NOTE: std:optional.value() fails to build on github runner
                // for arm64 macos although this function is on the
                // c++17 standard.
                // This alternative way to access the value works correctly.
                const auto cur_time = *cur_time_or;

                // Finish conditions
                if (end_t.has_value() &&
                    cur_time.seconds_since_epoch > end_t.value()) {
                  keep_looking = false;
                  break;
                }

                if (max_ticks > 0 && ticks.size() >= max_ticks) {
                  keep_looking = false;
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
              second += step;
            }
            second = min_second;
            minute += step;
          }
          second = min_second;
          minute = min_minute;
          hour += step;
        }
        second = min_second;
        minute = min_minute;
        hour = min_hour;
        mday += step;
      }
      second = min_second;
      minute = min_minute;
      hour = min_hour;
      mday = min_mday;
      month += step;
    }
    second = min_second;
    minute = min_minute;
    hour = min_hour;
    mday = min_mday;
    month = min_month;
    year += step;
  }
  return ticks;
}

py::array_t<double> tick_calendar(
    const double start_timestamp,                // min date
    const double end_timestamp,                  // max date
    const int min_second, const int max_second,  // second range
    const int min_minute, const int max_minute,  // minute range
    const int min_hour, const int max_hour,      // hours range
    const int min_mday, const int max_mday,      // month days
    const int min_month, const int max_month,    // month range
    const int min_wday, const int max_wday,      // weekdays
    const bool include_right, const bool include_left) {
  auto ticks =
      find_ticks(start_timestamp, end_timestamp, true, -1, min_second,
                 max_second, min_minute, max_minute, min_hour, max_hour,
                 min_mday, max_mday, min_month, max_month, min_wday, max_wday);

  if (include_right && (ticks.back() < end_timestamp)) {
    // starting from the end, find 1 tick to the right
    auto right_ticks =
        find_ticks(end_timestamp, std::nullopt, true, 1, min_second, max_second,
                   min_minute, max_minute, min_hour, max_hour, min_mday,
                   max_mday, min_month, max_month, min_wday, max_wday);
    ticks.insert(ticks.end(), right_ticks.begin(), right_ticks.end());
  }

  if (include_left && (ticks.front() > start_timestamp)) {
    // starting from the start, find 1 tick to the left
    auto left_ticks = find_ticks(start_timestamp, std::nullopt, false, 1,
                                 min_second, max_second, min_minute, max_minute,
                                 min_hour, max_hour, min_mday, max_mday,
                                 min_month, max_month, min_wday, max_wday);
    ticks.insert(ticks.begin(), left_ticks.begin(), left_ticks.end());
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
        py::arg("max_wday"), py::arg("include_right"), py::arg("include_left"));
}
