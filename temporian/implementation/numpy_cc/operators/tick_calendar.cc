#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

py::array_t<double> tick_calendar(
    const long start_timestamp,                  // min date
    const long end_timestamp,                    // max date
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
  std::tm start_utc = *std::gmtime(&start_timestamp);
  std::tm end_utc = *std::gmtime(&end_timestamp);

  int year = start_utc.tm_year;                           // from 1900
  int month = std::max(start_utc.tm_mon + 1, min_month);  // zero-based tm_mon
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
              std::tm tm_struct = {};
              tm_struct.tm_year = year;      // Since 1900
              tm_struct.tm_mon = month - 1;  // zero-based
              tm_struct.tm_mday = mday;
              tm_struct.tm_hour = hour;
              tm_struct.tm_min = minute;
              tm_struct.tm_sec = second;

              std::time_t time = std::mktime(&tm_struct);

              // Valid date
              if (time != -1 && tm_struct.tm_mday == mday) {
                // Finish condition
                if (time > end_timestamp) {
                  in_range = false;
                  break;
                }

                // Check weekday match
                if (tm_struct.tm_wday >= min_wday &&
                    tm_struct.tm_wday <= max_wday) {
                  ticks.push_back(time);
                }
              } else {
                // Invalid date (end of month)
                second = max_second;  // avoid unnecessary loops
                minute = max_minute;
                hour = max_hour;
              }
              second++;
            }
            second = min_second;
            minute++;
          }
          minute = min_minute;
          hour++;
        }
        hour = min_hour;
        mday++;
      }
      mday = min_mday;
      month++;
    }
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

void init_tick_calendar(py::module &m) {
  m.def("tick_calendar", &tick_calendar, "", py::arg("start_timestamp"),
        py::arg("end_timestamp"), py::arg("min_second"), py::arg("max_second"),
        py::arg("min_minute"), py::arg("max_minute"), py::arg("min_hour"),
        py::arg("max_hour"), py::arg("min_mday"), py::arg("max_mday"),
        py::arg("min_month"), py::arg("max_month"), py::arg("min_wday"),
        py::arg("max_wday"));
}
