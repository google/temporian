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
  const long start_t = (long)std::floor(start_timestamp);
  const long end_t = (long)std::floor(end_timestamp);

  std::tm start_utc = *std::gmtime(&start_t);

  int year = start_utc.tm_year;                           // from 1900
  int month = std::max(start_utc.tm_mon + 1, min_month);  // zero-based tm_mon
  int mday = std::max(start_utc.tm_mday, min_mday);       // 1-31
  int hour = std::max(start_utc.tm_hour, min_hour);
  int minute = std::max(start_utc.tm_min, min_minute);
  int second = std::max(start_utc.tm_sec, min_second);

  // Workaround to get timestamp from UTC datetimes (mktime depends on timezone)
  std::tm start_local = *std::localtime(&start_t);
  const int offset_tzone = std::mktime(&start_utc) - std::mktime(&start_local);

  bool in_range = true;
  while (in_range) {
    while (month <= max_month && in_range) {
      while (mday <= max_mday && in_range) {
        while (hour <= max_hour && in_range) {
          while (minute <= max_minute && in_range) {
            while (second <= max_second && in_range) {
              std::tm tm_date = {};
              tm_date.tm_year = year;      // Since 1900
              tm_date.tm_mon = month - 1;  // zero-based
              tm_date.tm_mday = mday;
              tm_date.tm_hour = hour;
              tm_date.tm_min = minute;
              tm_date.tm_sec = second;
              tm_date.tm_isdst = 0;
              tm_date.tm_gmtoff = start_local.tm_gmtoff;

              // This assumes that the date is in local timezone
              const std::time_t time_local = std::mktime(&tm_date);

              // Valid date
              if (time_local != -1 && tm_date.tm_mday == mday) {
                // Remove timezone offset from timestamp
                const std::time_t time_utc = time_local - offset_tzone;

                // Finish condition
                if (time_utc > end_t) {
                  in_range = false;
                  break;
                }

                // Check weekday match (mktime sets it properly)
                if (tm_date.tm_wday >= min_wday &&
                    tm_date.tm_wday <= max_wday) {
                  ticks.push_back(time_utc);
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

void init_tick_calendar(py::module &m) {
  m.def("tick_calendar", &tick_calendar, "", py::arg("start_timestamp"),
        py::arg("end_timestamp"), py::arg("min_second"), py::arg("max_second"),
        py::arg("min_minute"), py::arg("max_minute"), py::arg("min_hour"),
        py::arg("max_hour"), py::arg("min_mday"), py::arg("max_mday"),
        py::arg("min_month"), py::arg("max_month"), py::arg("min_wday"),
        py::arg("max_wday"));
}
