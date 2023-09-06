#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

py::array_t<double> tick_calendar(
    const double start_timestamp, const double end_timestamp,  // boundaries
    const int min_second, const int max_second,                // second range
    const int min_minute, const int max_minute,                // minute range
    const int min_hour, const int max_hour,                    // hours range
    const int min_mday, const int max_mday,                    // month days
    const int min_month, const int max_month,                  // month range
    const int min_wday, const int max_wday                     // weekdays
) {
  // Variable length ticks
  std::vector<double> ticks;

  int second = 0;

  while (second <= 10) {
    ticks.push_back(second);
    second++;
  }

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
